use crate::{exterior::ExteriorRank, mesh::SimplicialManifold, simplicial::CellComplex, whitney};

pub trait ElmatProvider {
  fn eval(&self, cell: &CellComplex) -> na::DMatrix<f64>;
}

impl<F> ElmatProvider for F
where
  F: Fn(&CellComplex) -> na::DMatrix<f64>,
{
  fn eval(&self, cell: &CellComplex) -> na::DMatrix<f64> {
    self(cell)
  }
}

pub trait ElvecProvider {
  fn eval(&self, cell: &CellComplex) -> na::DVector<f64>;
}
impl<F> ElvecProvider for F
where
  F: Fn(&CellComplex) -> na::DVector<f64>,
{
  fn eval(&self, cell: &CellComplex) -> nalgebra::DVector<f64> {
    self(cell)
  }
}

/// Exact Element Matrix Provider for the Laplace-Beltrami operator.
///
/// $A = [(dif lambda_tau, dif lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$
pub fn laplace_beltrami_elmat(cell: &CellComplex) -> na::DMatrix<f64> {
  let ref_difbarys = whitney::ref_difbarys(cell.dim());
  cell.vol() * cell.metric().covector_norm_sqr(&ref_difbarys)
}

/// Exact Element Matrix Provider for the exterior derivative part of Hodge-Laplace operator.
///
/// $A = [inner(dif lambda_tau, dif lambda_sigma)_(L^2 Lambda^(k+1) (K))]_(sigma,tau in Delta_k (K))$
pub fn hodge_laplace_dif_elmat(cell: &CellComplex, k: ExteriorRank) -> na::DMatrix<f64> {
  let ref_difwhitneys = whitney::ref_difwhitneys(cell.dim(), k);
  cell.vol() * cell.metric().kform_norm_sqr(k, &ref_difwhitneys)
}

/// Exact Element Matrix Provider for the codifferential part of Hodge-Laplace operator.
///
/// $A = [inner(delta lambda_tau, delta lambda_sigma)_(L^2 Lambda^(k-1) (K))]_(sigma,tau in Delta_k (K))$
pub fn hodge_laplace_codif_elmat(cell: &CellComplex, k: ExteriorRank) -> na::DMatrix<f64> {
  let ref_codifwhitneys = whitney::ref_codifwhitneys(cell.dim(), k);
  cell.vol() * cell.metric().kform_norm_sqr(k, &ref_codifwhitneys)
}

/// Exact Element Matrix Provider for the full Hodge-Laplace operator.
pub fn hodge_laplace_elmat(cell: &CellComplex, k: ExteriorRank) -> na::DMatrix<f64> {
  hodge_laplace_dif_elmat(cell, k) + hodge_laplace_codif_elmat(cell, k)
}

/// Exact Element Matrix Provider for scalar mass bilinear form.
pub fn mass_elmat(cell: &CellComplex) -> na::DMatrix<f64> {
  let ndofs = cell.nvertices();
  let dim = cell.dim();
  let v = cell.vol() / ((dim + 1) * (dim + 2)) as f64;
  let mut elmat = na::DMatrix::from_element(ndofs, ndofs, v);
  elmat.fill_diagonal(2.0 * v);
  elmat
}

/// Approximated Element Matrix Provider for scalar mass bilinear form,
/// obtained through trapezoidal quadrature rule.
pub fn lumped_mass_elmat(cell: &CellComplex) -> na::DMatrix<f64> {
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
  fn eval(&self, cell: &CellComplex) -> na::DVector<f64> {
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
  for cell in mesh.cells() {
    let mut sum = 0.0;
    for &ivertex in cell.oriented_vertplex().iter() {
      sum += fn_coeffs[ivertex].powi(2);
    }
    let nvertices = cell.nvertices();
    let cell_geo = cell.as_cell_complex();
    let vol = cell_geo.vol();
    norm += (vol / nvertices as f64) * sum;
  }
  norm.sqrt()
}

#[cfg(test)]
mod test {
  use super::{hodge_laplace_dif_elmat, laplace_beltrami_elmat};
  use crate::{linalg::assert_mat_eq, simplicial::ReferenceCell};

  #[test]
  fn hodge_laplace0_is_laplace_beltrami_refcell() {
    for n in 0..=3 {
      let cell = ReferenceCell::new(n).to_cell_complex();
      let laplace_beltrami = laplace_beltrami_elmat(&cell);
      let hodge_laplace = hodge_laplace_dif_elmat(&cell, 0);
      assert_mat_eq(&hodge_laplace, &laplace_beltrami);
    }
  }
}
