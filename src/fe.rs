use crate::{
  cell::{StandaloneCell, REFCELLS},
  combinatorics::generate_combinations,
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

/// $delta^k: cal(W) Lambda^k -> cal(W) Lambda^(k-1)$
/// Hodge adjoint of exterior derivative.
pub fn kcodifferential_local(cell: &StandaloneCell, k: Rank) -> na::DMatrix<f64> {
  let n = cell.dim();

  (-1f64).powi((n * (k + 1) + 1) as i32)
    * khodge_star_local(cell, n - k + 1)
    * kexterior_derivative_local(cell.dim(), n - k)
    * khodge_star_local(cell, k)
}

/// $star_k: cal(W) Lambda^k -> cal(W) Lambda^(n-k)$
pub fn khodge_star_local(_cell: &StandaloneCell, _k: Rank) -> na::DMatrix<f64> {
  todo!()
}

// TODO: Can we reasonably avoid the inverse?
// WARN: UNSTABLE
/// Inner product on covectors / 1-forms.
///
/// Represented as gram matrix on covector standard basis.
pub fn covector_gramian(cell: &StandaloneCell) -> na::DMatrix<f64> {
  let vector_gramian = cell.metric_tensor();
  vector_gramian.try_inverse().unwrap()
}

/// Inner product on k-forms
///
/// Represented as gram matrix on lexicographically ordered standard k-form standard basis.
pub fn kform_gramian(cell: &StandaloneCell, k: Rank) -> na::DMatrix<f64> {
  let n = cell.dim();
  let combinations = generate_combinations(n, k);
  let covector_gramian = covector_gramian(cell);

  let mut kform_gramian = na::DMatrix::zeros(combinations.len(), combinations.len());
  let mut kbasis_mat = na::DMatrix::zeros(k, k);

  for icomb in 0..combinations.len() {
    let combi = &combinations[icomb];
    for jcomb in icomb..combinations.len() {
      let combj = &combinations[jcomb];

      for iicomb in 0..k {
        let combii = combi[iicomb];
        for jjcomb in 0..k {
          let combjj = combj[jjcomb];
          kbasis_mat[(iicomb, jjcomb)] = covector_gramian[(combii, combjj)];
        }
      }
      let det = kbasis_mat.determinant();
      kform_gramian[(icomb, jcomb)] = det;
      kform_gramian[(jcomb, icomb)] = det;
    }
  }
  kform_gramian
}

/// Exact Element Matrix Provider for the negative Laplacian.
pub fn laplacian_neg_elmat(cell: &StandaloneCell) -> na::DMatrix<f64> {
  let dim = cell.dim();

  let mut reference_gradbarys = na::DMatrix::zeros(dim, dim + 1);
  for i in 0..dim {
    reference_gradbarys[(i, 0)] = -1.0;
  }
  for i in 0..dim {
    reference_gradbarys[(i, i + 1)] = 1.0;
  }

  let covector_gramian = covector_gramian(cell);
  cell.vol() * reference_gradbarys.transpose() * covector_gramian * reference_gradbarys
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

#[cfg(test)]
mod test {
  use num_integer::binomial;

  use crate::{cell::ReferenceCell, util::assert_mat_eq};

  use super::kform_gramian;

  #[test]
  fn kform_gramian_refcell() {
    for n in 0..=3 {
      let cell = ReferenceCell::new(n).to_standalone_cell();
      for k in 0..=n {
        let binom = binomial(n, k);
        let expected_gram = na::DMatrix::identity(binom, binom);
        let computed_gram = kform_gramian(&cell, k);
        assert_mat_eq(&computed_gram, &expected_gram);
      }
    }
  }
}
