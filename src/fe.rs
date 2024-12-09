use crate::{
  combo::{binomial, combinators::IndexSubsets, factorial, sort_signed, IndexSet},
  exterior::ExteriorRank,
  mesh::SimplicialManifold,
  simplicial::{CellComplex, REFCELLS},
  Dim,
};

use itertools::Itertools;

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

/// $dif^k: cal(W) Lambda^k -> cal(W) Lambda^(k+1)$
pub fn kexterior_derivative_local(cell_dim: Dim, k: ExteriorRank) -> na::DMatrix<f64> {
  REFCELLS[cell_dim].kboundary_operator(k + 1).transpose()
}

/// $delta^k: cal(W) Lambda^k -> cal(W) Lambda^(k-1)$
/// Hodge adjoint of exterior derivative.
pub fn kcodifferential_local(cell: &CellComplex, k: ExteriorRank) -> na::DMatrix<f64> {
  let n = cell.dim();

  (-1f64).powi((n * (k + 1) + 1) as i32)
    * khodge_star_local(cell, n - k + 1)
    * kexterior_derivative_local(cell.dim(), n - k)
    * khodge_star_local(cell, k)
}

/// $star_k: cal(W) Lambda^k -> cal(W) Lambda^(n-k)$
pub fn khodge_star_local(_cell: &CellComplex, _k: ExteriorRank) -> na::DMatrix<f64> {
  todo!()
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

/// The constant exterior derivatives of the reference Whitney forms, given in
/// the k-form standard basis.
pub fn ref_difwhitneys(n: Dim, k: ExteriorRank) -> na::DMatrix<f64> {
  let difk = k + 1;
  let difk_factorial = factorial(difk) as f64;

  let whitney_basis_size = binomial(n + 1, k + 1);
  let kform_basis_size = binomial(n, difk);

  let mut ref_difwhitneys = na::DMatrix::zeros(kform_basis_size, whitney_basis_size);
  for (whitney_comb_rank, whitney_comb) in IndexSubsets::canonical(n + 1, k + 1).enumerate() {
    if whitney_comb[0] == 0 {
      let kform_comb = whitney_comb.iter().skip(1).map(|c| *c - 1).collect_vec();
      for i in 0..n {
        let mut kform_comb = kform_comb.clone();
        kform_comb.insert(0, i);
        let sign = sort_signed(&mut kform_comb);

        kform_comb.dedup();
        if kform_comb.len() != difk {
          continue;
        };

        let kform_comb = IndexSet::from(kform_comb)
          .assume_sorted()
          .with_local_base(n);
        let kform_comb_rank = kform_comb.lex_rank();
        ref_difwhitneys[(kform_comb_rank, whitney_comb_rank)] = -sign.as_f64() * difk_factorial;
      }
    } else {
      let kform_comb = whitney_comb.iter().map(|c| *c - 1).collect_vec();
      let kform_comb = IndexSet::from(kform_comb)
        .assume_sorted()
        .with_local_base(n);
      let kform_comb_rank = kform_comb.lex_rank();
      ref_difwhitneys[(kform_comb_rank, whitney_comb_rank)] = difk_factorial;
    }
  }
  ref_difwhitneys
}

/// Exact Element Matrix Provider for the Laplace-Beltrami operator.
///
/// $A = [(dif lambda_tau, dif lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$
pub fn laplace_beltrami_elmat(cell: &CellComplex) -> na::DMatrix<f64> {
  let ref_difbarys = ref_difbarys(cell.dim());
  let covector_gramian = cell.metric().covector_gramian();
  cell.vol() * ref_difbarys.transpose() * covector_gramian * ref_difbarys
}

/// Exact Element Matrix Provider for the exterior derivative part of Hodge-Laplace operator.
///
/// $A = [inner(dif lambda_tau, dif lambda_sigma)_(L^2 Lambda^(k+1) (K))]_(sigma,tau in Delta_k (K))$
pub fn hodge_laplace_dif_elmat(cell: &CellComplex, k: ExteriorRank) -> na::DMatrix<f64> {
  let ref_difwhitneys = ref_difwhitneys(cell.dim(), k);
  let form_gramian = cell.metric().kform_gramian(k + 1);
  cell.vol() * ref_difwhitneys.transpose() * form_gramian * ref_difwhitneys
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

pub fn ref_bary(n: Dim, ibary: usize, x: na::DVector<f64>) -> f64 {
  assert!(ibary < n + 1);
  assert!(x.nrows() == n);

  if ibary == 0 {
    1.0 - x.sum()
  } else {
    x[ibary - 1]
  }
}

//pub fn ref_whitney(
//  n: Dim,
//  k: ExteriorRank,
//  iwhitney: &[usize],
//  x: na::DVector<f64>,
//) -> na::DVector<f64> {
//  let form_basis_size = binomial(n, k);
//  let coeffs = na::DVector::zeros(form_basis_size);
//  for l in 0..=k {
//    let coeff = Sign::from_parity(l).as_f64() * ref_bary(n, l, x);
//  }
//  factorial(k) as f64 * coeffs
//}

#[cfg(test)]
mod test {
  use super::{hodge_laplace_dif_elmat, laplace_beltrami_elmat, ref_difbarys, ref_difwhitneys};
  use crate::{linalg::assert_mat_eq, simplicial::ReferenceCell};

  use num_integer::binomial;

  #[test]
  fn ref_difwhitney0_is_ref_difbary() {
    for n in 0..=5 {
      let whitneys = ref_difwhitneys(n, 0);
      let barys = ref_difbarys(n);
      assert_mat_eq(&whitneys, &barys)
    }
  }
  #[test]
  fn ref_difwhitneyn_is_zero() {
    for n in 0..=5 {
      println!("n={n}");
      let whitneys = ref_difwhitneys(n, n);
      let zero = na::DMatrix::zeros(0, 1);
      assert_mat_eq(&whitneys, &zero)
    }
  }

  #[test]
  fn kform_gramian_refcell() {
    for n in 0..=3 {
      let cell = ReferenceCell::new(n).to_cell_complex();
      for k in 0..=n {
        let binom = binomial(n, k);
        let expected_gram = na::DMatrix::identity(binom, binom);
        let computed_gram = cell.metric().kform_gramian(k);
        assert_mat_eq(&computed_gram, &expected_gram);
      }
    }
  }

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
