use crate::{
  cell::StandaloneCell,
  mesh::{CellIdx, SimplicialManifold},
  space::FeSpace,
};

pub trait ElmatProvider {
  fn eval(&self, space: &FeSpace, icell: CellIdx) -> na::DMatrix<f64>;
}

impl<F> ElmatProvider for F
where
  F: Fn(&FeSpace, CellIdx) -> na::DMatrix<f64>,
{
  fn eval(&self, space: &FeSpace, icell: CellIdx) -> na::DMatrix<f64> {
    self(space, icell)
  }
}

pub trait ElvecProvider {
  fn eval(&self, space: &FeSpace, icell: CellIdx) -> na::DVector<f64>;
}
impl<F> ElvecProvider for F
where
  F: Fn(&FeSpace, CellIdx) -> na::DVector<f64>,
{
  fn eval(&self, space: &FeSpace, icell: CellIdx) -> nalgebra::DVector<f64> {
    self(space, icell)
  }
}

pub fn laplacian_neg_elmat_geo(cell_geo: &StandaloneCell) -> na::DMatrix<f64> {
  let dim = cell_geo.dim();
  let metric = cell_geo.metric_tensor();
  let det = cell_geo.det();

  let mut reference_gradbarys = na::DMatrix::zeros(dim, dim + 1);
  for i in 0..dim {
    reference_gradbarys[(i, 0)] = -1.0;
  }
  for i in 0..dim {
    reference_gradbarys[(i, i + 1)] = 1.0;
  }

  det * reference_gradbarys.transpose() * metric.lu().solve(&reference_gradbarys).unwrap()
}

/// Exact Element Matrix Provider for the negative Laplacian.
pub fn laplacian_neg_elmat(space: &FeSpace, icell: CellIdx) -> na::DMatrix<f64> {
  let cell_geo = space.mesh().cells().get_kidx(icell).as_standalone_cell();
  laplacian_neg_elmat_geo(&cell_geo)
}

/// Approximated Element Matrix Provider for mass bilinear form,
/// obtained through trapezoidal quadrature rule.
pub fn lumped_mass_elmat(space: &FeSpace, icell: CellIdx) -> na::DMatrix<f64> {
  let cell_geo = space.mesh().cells().get_kidx(icell).as_standalone_cell();
  let n = cell_geo.nvertices();
  let v = cell_geo.vol() / n as f64;
  na::DMatrix::from_diagonal_element(n, n, v)
}
/// Element Vector Provider for scalar load function.
///
/// Computed using trapezoidal quadrature rule.
/// Exact for constant load.
pub struct LoadElvec {
  dof_data: na::DVector<f64>,
}
impl LoadElvec {
  pub fn new(dof_data: na::DVector<f64>) -> Self {
    Self { dof_data }
  }
}
impl ElvecProvider for LoadElvec {
  fn eval(&self, space: &FeSpace, icell: CellIdx) -> na::DVector<f64> {
    let cell = space.mesh().cells().get_kidx(icell);
    let cell_geo = cell.as_standalone_cell();
    let nverts = cell_geo.nvertices();

    cell_geo.det() / nverts as f64
      * na::DVector::from_iterator(
        nverts,
        cell
          .ordered_vertplex()
          .iter()
          .copied()
          .map(|iv| self.dof_data[iv]),
      )
  }
}

pub fn l2_norm(fn_coeffs: na::DVector<f64>, mesh: &SimplicialManifold) -> f64 {
  let mut norm: f64 = 0.0;
  for cell in mesh.cells().iter() {
    let mut sum = 0.0;
    for &ivertex in cell.ordered_vertplex().iter() {
      sum += fn_coeffs[ivertex].powi(2);
    }
    let nvertices = cell.nvertices();
    let cell_geo = cell.as_standalone_cell();
    let vol = cell_geo.vol();
    norm += (vol / nvertices as f64) * sum;
  }
  norm.sqrt()
}
