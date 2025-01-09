pub mod whitney;

use exterior::ExteriorRank;
use geometry::metric::manifold::{local::LocalMetricComplex, MetricComplex};
use topology::Dim;

pub type DofIdx = usize;
pub type DofCoeff = f64;

// TODO: turn into cochain
pub type FeFunction = na::DVector<f64>;

pub type ElMat = na::DMatrix<f64>;
pub trait ElMatProvider {
  fn row_rank(&self) -> ExteriorRank;
  fn col_rank(&self) -> ExteriorRank;
  fn eval(&self, local_complex: &LocalMetricComplex) -> ElMat;
}

pub type ElVec = na::DVector<f64>;
pub trait ElVecProvider {
  fn rank(&self) -> ExteriorRank;
  fn eval(&self, local_complex: &LocalMetricComplex) -> ElVec;
}

/// Exact Element Matrix Provider for the Laplace-Beltrami operator.
///
/// $A = [(dif lambda_tau, dif lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$
pub struct LaplaceBeltramiElmat;
impl ElMatProvider for LaplaceBeltramiElmat {
  fn row_rank(&self) -> ExteriorRank {
    0
  }
  fn col_rank(&self) -> ExteriorRank {
    0
  }
  fn eval(&self, local_complex: &LocalMetricComplex) -> ElMat {
    let ref_difbarys = ref_difbarys(local_complex.dim());
    local_complex.vol() * local_complex.metric().covector_norm_sqr(&ref_difbarys)
  }
}

/// Exact Element Matrix Provider for scalar mass bilinear form.
pub struct ScalarMassElmat;
impl ElMatProvider for ScalarMassElmat {
  fn row_rank(&self) -> ExteriorRank {
    0
  }
  fn col_rank(&self) -> ExteriorRank {
    0
  }
  fn eval(&self, local_complex: &LocalMetricComplex) -> ElMat {
    let ndofs = local_complex.topology().nvertices();
    let dim = local_complex.dim();
    let v = local_complex.vol() / ((dim + 1) * (dim + 2)) as f64;
    let mut elmat = na::DMatrix::from_element(ndofs, ndofs, v);
    elmat.fill_diagonal(2.0 * v);
    elmat
  }
}

/// Approximated Element Matrix Provider for scalar mass bilinear form,
/// obtained through trapezoidal quadrature rule.
pub struct ScalarLumpedMassElmat;
impl ElMatProvider for ScalarLumpedMassElmat {
  fn row_rank(&self) -> ExteriorRank {
    0
  }
  fn col_rank(&self) -> ExteriorRank {
    0
  }
  fn eval(&self, local_complex: &LocalMetricComplex) -> ElMat {
    let n = local_complex.topology().nvertices();
    let v = local_complex.vol() / n as f64;
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
impl ElVecProvider for SourceElvec {
  fn rank(&self) -> ExteriorRank {
    0
  }
  fn eval(&self, local_complex: &LocalMetricComplex) -> ElVec {
    let nverts = local_complex.topology().nvertices();

    local_complex.vol() / nverts as f64
      * na::DVector::from_iterator(
        nverts,
        local_complex
          .topology()
          .vertices()
          .iter()
          .map(|&iv| self.dof_data[iv]),
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

pub fn l2_inner(a: &FeFunction, b: &FeFunction, mesh: &MetricComplex) -> f64 {
  integrate_pointwise(&a.component_mul(b), mesh)
}
pub fn l2_norm(a: &FeFunction, mesh: &MetricComplex) -> f64 {
  l2_inner(a, a, mesh)
}

// this is weird...
pub fn integrate_pointwise(func: &FeFunction, mesh: &MetricComplex) -> f64 {
  let mut norm: f64 = 0.0;
  for facet in mesh.topology().facets().iter() {
    let mut sum = 0.0;
    for vertex in facet.vertices() {
      sum += func[vertex.kidx()];
    }
    let nvertices = facet.nvertices();
    let vol = mesh.local_complex(facet).vol();
    norm += (vol / nvertices as f64) * sum;
  }
  norm.sqrt()
}
