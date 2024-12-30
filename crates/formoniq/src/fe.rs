pub mod whitney;

use common::Dim;
use exterior::ExteriorRank;
use manifold::{complex::KSimplexIdx, simplicial::LocalComplex, RiemannianComplex};

pub type DofIdx = KSimplexIdx;

pub trait ElmatProvider {
  fn row_rank(&self) -> ExteriorRank;
  fn col_rank(&self) -> ExteriorRank;
  fn eval(&self, cell: &LocalComplex) -> na::DMatrix<f64>;
}

pub trait ElvecProvider {
  fn rank(&self) -> ExteriorRank;
  fn eval(&self, cell: &LocalComplex) -> na::DVector<f64>;
}

/// Exact Element Matrix Provider for the Laplace-Beltrami operator.
///
/// $A = [(dif lambda_tau, dif lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$
pub struct LaplaceBeltramiElmat;
impl ElmatProvider for LaplaceBeltramiElmat {
  fn row_rank(&self) -> ExteriorRank {
    0
  }
  fn col_rank(&self) -> ExteriorRank {
    0
  }
  fn eval(&self, cell: &LocalComplex) -> na::DMatrix<f64> {
    let ref_difbarys = ref_difbarys(cell.dim());
    cell.vol() * cell.metric().covector_norm_sqr(&ref_difbarys)
  }
}

/// Exact Element Matrix Provider for scalar mass bilinear form.
pub struct ScalarMassElmat;
impl ElmatProvider for ScalarMassElmat {
  fn row_rank(&self) -> ExteriorRank {
    0
  }
  fn col_rank(&self) -> ExteriorRank {
    0
  }
  fn eval(&self, cell: &LocalComplex) -> na::DMatrix<f64> {
    let ndofs = cell.nvertices();
    let dim = cell.dim();
    let v = cell.vol() / ((dim + 1) * (dim + 2)) as f64;
    let mut elmat = na::DMatrix::from_element(ndofs, ndofs, v);
    elmat.fill_diagonal(2.0 * v);
    elmat
  }
}

/// Approximated Element Matrix Provider for scalar mass bilinear form,
/// obtained through trapezoidal quadrature rule.
pub struct ScalarLumpedMassElmat;
impl ElmatProvider for ScalarLumpedMassElmat {
  fn row_rank(&self) -> ExteriorRank {
    0
  }
  fn col_rank(&self) -> ExteriorRank {
    0
  }
  fn eval(&self, cell: &LocalComplex) -> na::DMatrix<f64> {
    let n = cell.nvertices();
    let v = cell.vol() / n as f64;
    na::DMatrix::from_diagonal_element(n, n, v)
  }
}

/// Element Vector Provider for scalar source function.
///
/// Computed using trapezoidal quadrature rule.
/// Exact for constant source.
pub struct SourceElvec {
  dof_data: na::DVector<f64>,
}
impl SourceElvec {
  pub fn new(dof_data: na::DVector<f64>) -> Self {
    Self { dof_data }
  }
}
impl ElvecProvider for SourceElvec {
  fn rank(&self) -> ExteriorRank {
    0
  }
  fn eval(&self, cell: &LocalComplex) -> na::DVector<f64> {
    let nverts = cell.nvertices();

    cell.vol() / nverts as f64
      * na::DVector::from_iterator(
        nverts,
        cell.vertices().iter().copied().map(|iv| self.dof_data[iv]),
      )
  }
}

/// The constant exterior drivatives of the reference barycentric coordinate
/// functions, given in the 1-form standard basis.
pub fn ref_difbarys(n: Dim) -> na::DMatrix<f64> {
  let mut ref_difbarys = na::DMatrix::zeros(n, n + 1);
  for i in 0..n {
    ref_difbarys[(i, 0)] = -1.0;
    ref_difbarys[(i, i + 1)] = 1.0;
  }
  ref_difbarys
}

pub fn integrate_pointwise<'a>(
  a: impl Into<na::DVectorView<'a, f64>>,
  mesh: &RiemannianComplex,
) -> f64 {
  let a = a.into();

  let mut norm: f64 = 0.0;
  for cell in mesh.cells() {
    let mut sum = 0.0;
    for &ivertex in cell.oriented_vertplex().iter() {
      sum += a[ivertex];
    }
    let nvertices = cell.nvertices();
    let cell_geo = cell.as_cell_complex();
    let vol = cell_geo.vol();
    norm += (vol / nvertices as f64) * sum;
  }
  norm.sqrt()
}
pub fn l2_inner<'a>(
  a: impl Into<na::DVectorView<'a, f64>>,
  b: impl Into<na::DVectorView<'a, f64>>,
  mesh: &RiemannianComplex,
) -> f64 {
  let a = a.into();
  let b = b.into();
  integrate_pointwise(&a.component_mul(&b), mesh)
}
pub fn l2_norm<'a>(a: impl Into<na::DVectorView<'a, f64>>, mesh: &RiemannianComplex) -> f64 {
  let a = a.into();
  l2_inner(a, a, mesh)
}
