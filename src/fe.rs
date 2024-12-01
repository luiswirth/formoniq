use crate::{
  cell::{StandaloneCell, REFCELLS},
  mesh::SimplicialManifold,
  Dim, Rank,
};

pub trait ElmatProvider {
  fn eval(&self, cell: &StandaloneCell) -> na::DMatrix<f64>;
}

impl<F> ElmatProvider for F
where
  F: Fn(&StandaloneCell) -> na::DMatrix<f64>,
{
  fn eval(&self, cell: &StandaloneCell) -> na::DMatrix<f64> {
    self(cell)
  }
}

pub trait ElvecProvider {
  fn eval(&self, cell: &StandaloneCell) -> na::DVector<f64>;
}
impl<F> ElvecProvider for F
where
  F: Fn(&StandaloneCell) -> na::DVector<f64>,
{
  fn eval(&self, cell: &StandaloneCell) -> nalgebra::DVector<f64> {
    self(cell)
  }
}

/// $dif^k: cal(W) Lambda^k -> cal(W) Lambda^(k+1)$
pub fn kexterior_derivative_local(cell_dim: Dim, k: Rank) -> na::DMatrix<f64> {
  REFCELLS[cell_dim].kboundary_operator(k + 1).transpose()
}

/// $star_k: cal(W) Lambda^k -> cal(W) Lambda^(n-k)$
pub fn hodge_star(_cell: &StandaloneCell, _k: Rank) -> na::DMatrix<f64> {
  // solve LSE involving metric tensor
  todo!()
}

/// Exact Element Matrix Provider for the negative Laplacian.
pub fn laplacian_neg_elmat(cell: &StandaloneCell) -> na::DMatrix<f64> {
  let dim = cell.dim();
  let metric = cell.metric_tensor();
  let det = cell.vol();

  let mut reference_gradbarys = na::DMatrix::zeros(dim, dim + 1);
  for i in 0..dim {
    reference_gradbarys[(i, 0)] = -1.0;
  }
  for i in 0..dim {
    reference_gradbarys[(i, i + 1)] = 1.0;
  }

  det * reference_gradbarys.transpose() * metric.lu().solve(&reference_gradbarys).unwrap()
}

/// Exact Element Matrix Provider for mass bilinear form.
pub fn mass_elmat(cell: &StandaloneCell) -> na::DMatrix<f64> {
  let ndofs = cell.nvertices();
  let dim = cell.dim();
  let v = cell.vol() / ((dim + 1) * (dim + 2)) as f64;
  let mut elmat = na::DMatrix::from_element(ndofs, ndofs, v);
  elmat.fill_diagonal(2.0 * v);
  elmat
}

/// Approximated Element Matrix Provider for mass bilinear form,
/// obtained through trapezoidal quadrature rule.
pub fn lumped_mass_elmat(cell: &StandaloneCell) -> na::DMatrix<f64> {
  let n = cell.nvertices();
  let v = cell.vol() / n as f64;
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
  fn eval(&self, cell: &StandaloneCell) -> na::DVector<f64> {
    let nverts = cell.nvertices();

    cell.vol() / nverts as f64
      * na::DVector::from_iterator(
        nverts,
        cell.vertices().iter().copied().map(|iv| self.dof_data[iv]),
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
