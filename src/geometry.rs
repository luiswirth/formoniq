use std::rc::Rc;

use crate::{
  combinatorics::factorial,
  mesh::{MeshNodes, SimplicialMesh},
  orientation::Orientation,
  util::gram_det_sqrt,
  Dim,
};

#[derive(Debug, Clone)]
pub struct GeometrySimplex {
  vertices: na::DMatrix<f64>,
}
impl GeometrySimplex {
  pub fn new(vertices: na::DMatrix<f64>) -> Self {
    assert!(!vertices.is_empty());
    Self { vertices }
  }

  /// Constructs a reference simplex in `dim` dimensions.
  /// The simplex has unit vertices plus the origin.
  /// The ambient and intrinsic dimension are the same.
  pub fn new_ref(dim: Dim) -> Self {
    Self::new_ref_embedded(dim, dim)
  }

  pub fn new_ref_embedded(intrinsic_dim: Dim, ambient_dim: Dim) -> Self {
    assert!(intrinsic_dim <= ambient_dim);
    let mut vertices = na::DMatrix::zeros(ambient_dim, intrinsic_dim + 1);
    // first col is already all zeros (origin)
    for d in 0..intrinsic_dim {
      vertices[(d, d + 1)] = 1.0;
    }
    Self { vertices }
  }

  pub fn dim_intrinsic(&self) -> Dim {
    self.vertices.ncols() - 1
  }
  pub fn dim_ambient(&self) -> Dim {
    self.vertices.nrows()
  }
  pub fn dims_agree(&self) -> bool {
    self.dim_intrinsic() == self.dim_ambient()
  }

  pub fn nvertices(&self) -> usize {
    self.vertices.ncols()
  }
  pub fn vertices(&self) -> &na::DMatrix<f64> {
    &self.vertices
  }
  pub fn vertex(&self, i: usize) -> na::DVectorView<f64> {
    self.vertices.column(i)
  }

  /// The vectors you get by subtracing a reference point (here the first one),
  /// from all other points.
  /// These vectors are then the axes of the simplex.
  pub fn spanning_vectors(&self) -> na::DMatrix<f64> {
    let p = self.vertex(0);
    let n = self.nvertices() - 1;
    let mut mat = na::DMatrix::zeros(self.dim_ambient(), n);
    for (c, v) in self.vertices().column_iter().skip(1).enumerate() {
      mat.column_mut(c).copy_from(&(v - p));
    }
    mat
  }

  /// Transformation from reference simplex to this simplex.
  pub fn reference_transform(&self) -> AffineTransform {
    let linear = self.spanning_vectors();
    println!("{:?}", linear.shape());
    let translation = self.vertex(0).into();
    AffineTransform {
      linear,
      translation,
    }
  }

  /// The determinate (signed volume) of the simplex.
  pub fn det(&self) -> f64 {
    let mat = self.spanning_vectors();
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

  /// The orientation of the simplex.
  pub fn orientation(&self) -> Orientation {
    match self.det().is_sign_positive() {
      true => Orientation::Pos,
      false => Orientation::Neg,
    }
  }

  /// The diameter of the simplex.
  /// This is the maximum distance of two points inside the simplex.
  pub fn diameter(&self) -> f64 {
    let n = self.nvertices();
    let mut dia = 0.0;

    for i in 0..n {
      for j in (i + 1)..n {
        let dist = (self.vertices().column(i) - self.vertices().column(j)).norm();
        if dist > dia {
          dia = dist;
        }
      }
    }
    dia
  }

  /// The shape regualrity measure of the simplex.
  pub fn shape_reguarity_measure(&self) -> f64 {
    self.diameter().powi(self.dim_intrinsic() as i32) / self.vol()
  }

  /// Linear map from $[lambda_1, dots lambda_k]^T -> [1, x_1 dots, x_n]^T$,
  /// where $k$ is the intrinsic dim and $n$ is the ambient dim.
  /// Maps barycentric coordinates to cartesian coordinates.
  pub fn barycentric_to_cartesian_map(&self) -> na::DMatrix<f64> {
    self.vertices.clone().insert_row(0, 1.0)
  }

  /// Linear map from $[1, x_1 dots, x_n]^T -> [lambda_1, dots lambda_k]^T$,
  /// where $k$ is the intrinsic dim and $n$ is the ambient dim.
  /// Maps cartesian coordinates to barycentric coordinates.
  pub fn cartesian_to_barycentric_map(&self) -> na::DMatrix<f64> {
    let m = self.barycentric_to_cartesian_map();
    if self.dims_agree() {
      m.try_inverse().unwrap()
    } else {
      m.pseudo_inverse(1e-10).unwrap()
    }
  }

  /// Constant gradients of barycentric coordinate functions.
  pub fn barycentric_functions_grad(&self) -> na::DMatrix<f64> {
    let n = self.nvertices();
    self
      .cartesian_to_barycentric_map()
      .transpose()
      .view_range(1..n, 0..n)
      .clone_owned()
  }

  /// The unnormalized normal vectors of the faces ((d-1)-simplicies).
  /// The ordering of these normal vectors corresponds to the natural ordering of
  /// the subsimplicies, given by removing one vertex at a time in order.
  pub fn face_normals(&self) -> na::DMatrix<f64> {
    -self.barycentric_functions_grad()
  }

  /// The normalized normal vectors of the faces ((d-1)-simplicies).
  pub fn face_unit_normals(&self) -> na::DMatrix<f64> {
    self.face_normals().normalize()
  }

  pub fn into_singleton_mesh(self) -> Rc<SimplicialMesh> {
    let nvertices = self.nvertices();
    let nodes = MeshNodes::new(self.vertices);
    SimplicialMesh::from_cells(nodes, vec![(0..nvertices).collect()])
  }
}

pub fn ref_vol(dim: Dim) -> f64 {
  (factorial(dim) as f64).recip()
}

pub struct AffineTransform {
  linear: na::DMatrix<f64>,
  translation: na::DVector<f64>,
}
impl AffineTransform {
  pub fn apply(&self, p: na::DMatrixView<f64>) -> na::DMatrix<f64> {
    let mut r = &self.linear * p;
    r.column_iter_mut().for_each(|mut c| c += &self.translation);
    r
  }
  pub fn det(&self) -> f64 {
    self.linear.determinant()
  }
  pub fn jacobi(&self) -> &na::DMatrix<f64> {
    &self.linear
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn ref_vol_test() {
    for d in 0..=8 {
      let simp = GeometrySimplex::new_ref(d);
      assert_eq!(simp.det(), ref_vol(d));
    }
  }

  #[test]
  fn reference_transform() {
    let refsimp = GeometrySimplex::new_ref(3);
    let simp = GeometrySimplex::new(na::DMatrix::from_columns(&[
      na::DVector::from_row_slice(&[5.0, 6.0, -2.0]),
      na::DVector::from_row_slice(&[2.0, 10.0, -1.0]),
      na::DVector::from_row_slice(&[1.0, 3.0, -3.0]),
      na::DVector::from_row_slice(&[0.0, -2.0, -4.0]),
    ]));
    println!("ref{:?}", refsimp.vertices().shape());
    assert_eq!(
      simp
        .reference_transform()
        .apply(refsimp.vertices().as_view()),
      *simp.vertices()
    );
    assert_eq!(simp.det(), refsimp.det() * simp.reference_transform().det());
  }
}
