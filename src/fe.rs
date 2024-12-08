use itertools::Itertools;
use num_integer::binomial;

use crate::{
  combo::{combinators::IndexSubsets, exterior::ExteriorRank, factorial, sort_signed, IndexSet},
  mesh::SimplicialManifold,
  simplicial::{CellComplex, REFCELLS},
  Dim,
};

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

// TODO: Can we reasonably avoid the inverse?
// WARN: UNSTABLE
/// Inner product on covectors / 1-forms.
///
/// Represented as gram matrix on covector standard basis.
pub fn covector_gramian(cell: &CellComplex) -> na::DMatrix<f64> {
  let vector_gramian = cell.metric_tensor();
  vector_gramian.try_inverse().unwrap()
}

/// Inner product on k-forms
///
/// Represented as gram matrix on lexicographically ordered standard k-form standard basis.
pub fn kform_gramian(cell: &CellComplex, k: ExteriorRank) -> na::DMatrix<f64> {
  let n = cell.dim();
  let combinations: Vec<_> = IndexSubsets::canonical(n, k).collect();
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
  let covector_gramian = covector_gramian(cell);
  cell.vol() * ref_difbarys.transpose() * covector_gramian * ref_difbarys
}

/// Exact Element Matrix Provider for the exterior derivative part of Hodge-Laplace operator.
///
/// $A = [inner(dif lambda_tau, dif lambda_sigma)_(L^2 Lambda^(k+1) (K))]_(sigma,tau in Delta_k (K))$
pub fn hodge_laplace_dif_elmat(cell: &CellComplex, k: ExteriorRank) -> na::DMatrix<f64> {
  let ref_difwhitneys = ref_difwhitneys(cell.dim(), k);
  let kform_gramian = kform_gramian(cell, k);
  cell.vol() * ref_difwhitneys.transpose() * kform_gramian * ref_difwhitneys
}

/// Exact Element Matrix Provider for mass bilinear form.
pub fn mass_elmat(cell: &CellComplex) -> na::DMatrix<f64> {
  let ndofs = cell.nvertices();
  let dim = cell.dim();
  let v = cell.vol() / ((dim + 1) * (dim + 2)) as f64;
  let mut elmat = na::DMatrix::from_element(ndofs, ndofs, v);
  elmat.fill_diagonal(2.0 * v);
  elmat
}

/// Approximated Element Matrix Provider for mass bilinear form,
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
  use super::{kform_gramian, ref_difbarys, ref_difwhitneys};
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
        let computed_gram = kform_gramian(&cell, k);
        assert_mat_eq(&computed_gram, &expected_gram);
      }
    }
  }
}
