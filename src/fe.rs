use crate::{
  mesh::{data::NodeData, CellId, SimplicialManifold},
  space::FeSpace,
};

pub trait ElmatProvider {
  fn eval(&self, space: &FeSpace, icell: CellId) -> na::DMatrix<f64>;
}

impl<F> ElmatProvider for F
where
  F: Fn(&FeSpace, CellId) -> na::DMatrix<f64>,
{
  fn eval(&self, space: &FeSpace, icell: CellId) -> na::DMatrix<f64> {
    self(space, icell)
  }
}

pub trait ElvecProvider {
  fn eval(&self, space: &FeSpace, icell: CellId) -> na::DVector<f64>;
}
impl<F> ElvecProvider for F
where
  F: Fn(&FeSpace, CellId) -> na::DVector<f64>,
{
  fn eval(&self, space: &FeSpace, icell: CellId) -> nalgebra::DVector<f64> {
    self(space, icell)
  }
}

/// The exact Element Matrix for the negative laplacian in linear lagrangian FE.
pub fn laplacian_neg_elmat(space: &FeSpace, icell: CellId) -> na::DMatrix<f64> {
  let cell_geo = space.mesh().cell(icell).geometry_simplex();
  let dim = cell_geo.dim();
  let mut reference_gradbarys = na::DMatrix::zeros(dim, dim + 1);
  for i in 0..dim {
    reference_gradbarys[(i, 0)] = -1.0;
  }
  for i in 0..dim {
    reference_gradbarys[(i, i + 1)] = 1.0;
  }
  let metric = cell_geo.metric_tensor();
  let vol = cell_geo.vol();

  vol * reference_gradbarys.transpose() * metric.lu().solve(&reference_gradbarys).unwrap()
}

/// Approximated Element Matrix for mass bilinear form,
/// obtained through trapezoidal quadrature rule.
pub fn lumped_mass_elmat(space: &FeSpace, icell: CellId) -> na::DMatrix<f64> {
  let cell_geo = space.mesh().cell(icell).geometry_simplex();
  let n = cell_geo.nvertices();
  let v = cell_geo.vol() / n as f64;
  na::DMatrix::from_diagonal_element(n, n, v)
}

// Element vector for scalar load function, computed using trapezoidal rule.
pub struct LoadElvec {
  dof_data: NodeData<f64>,
}
impl ElvecProvider for LoadElvec {
  fn eval(&self, space: &FeSpace, icell: CellId) -> na::DVector<f64> {
    let cell = space.mesh().cell(icell);
    let cell_geo = cell.geometry_simplex();
    let nverts = cell_geo.nvertices();
    cell_geo.vol() / nverts as f64
      * na::DVector::from_iterator(
        nverts,
        cell.vertices().iter().copied().map(|iv| self.dof_data[iv]),
      )
  }
}
impl LoadElvec {
  pub fn new(dof_data: NodeData<f64>) -> Self {
    Self { dof_data }
  }
}

// NOTE: In general this should depend on the FE Space and not just the mesh.
pub fn l2_norm(fn_coeffs: na::DVector<f64>, mesh: &SimplicialManifold) -> f64 {
  let mut norm = 0.0;
  for (icell, cell) in mesh.cells().iter().enumerate() {
    let mut sum = 0.0;
    for &ivertex in cell.vertices() {
      sum += fn_coeffs[ivertex].powi(2);
    }
    let nvertices = cell.nvertices();
    let vol = mesh.cell(icell).geometry_simplex().vol();
    norm += (vol / nvertices as f64) * sum;
  }
  norm.sqrt()
}

#[cfg(test)]
mod test {
  use super::laplacian_neg_elmat;
  use crate::{geometry::GeometrySimplex, space::FeSpace};

  fn check_elmat_refd(d: usize, expected_elmat: na::DMatrixView<f64>) {
    let space = FeSpace::new(GeometrySimplex::new_ref(d).into_singleton_mesh());
    let computed_elmat = laplacian_neg_elmat(&space, 0);
    assert!((computed_elmat - expected_elmat).norm() < 10.0 * f64::EPSILON);
  }

  #[test]
  fn laplacian_elmat_ref1d() {
    #[rustfmt::skip]
    let expected_elmat = na::DMatrix::from_row_slice(2, 2, &[
      1.0, -1.0,
      -1.0, 1.0
    ]);
    check_elmat_refd(1, expected_elmat.as_view());
  }

  #[test]
  fn laplacian_elmat_ref2d() {
    #[rustfmt::skip]
    let expected_elmat = na::DMatrix::from_row_slice(3, 3, &[
      1.0, -0.5, -0.5,
      -0.5, 0.5, 0.0,
      -0.5, 0.0, 0.5
    ]);
    check_elmat_refd(2, expected_elmat.as_view());
  }

  #[test]
  fn laplacian_elmat_ref3d() {
    let a = 1. / 6.;
    #[rustfmt::skip]
    let expected_elmat = na::DMatrix::from_row_slice(4, 4, &[
      0.5, -a, -a, -a,
      -a, a, 0., 0.,
      -a, 0., a, 0.,
      -a, 0., 0., a
    ]);
    check_elmat_refd(3, expected_elmat.as_view());
  }
}
