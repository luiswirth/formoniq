use common::Dim;
use exterior::{ExteriorRank, RiemannianMetricExt};
use index_algebra::{
  binomial,
  combinators::IndexSubsets,
  factorial,
  sign::{sort_signed, Sign},
  variants::*,
  IndexAlgebra,
};
use manifold::simplicial::{LocalComplex, REFCELLS};

use super::scalar_mass_elmat;

/// $dif^k: cal(W) Lambda^k -> cal(W) Lambda^(k+1)$
pub fn exterior_derivative(cell_dim: Dim, k: ExteriorRank) -> na::DMatrix<f64> {
  REFCELLS[cell_dim].kboundary_operator(k + 1).transpose()
}

/// Element Matrix for the weak Hodge star operator / the mass bilinear form on the reference element.
///
/// $M = [inner(star lambda_tau, lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$
pub fn hodge_mass_elmat(cell: &LocalComplex, k: ExteriorRank) -> na::DMatrix<f64> {
  let n = cell.dim();
  let difk = k + 1;
  let k_factorial = factorial(k) as f64;
  let k_factorial_sqr = k_factorial.powi(2);

  let kwhitney_basis_size = binomial(n + 1, k + 1);

  let scalar_mass = scalar_mass_elmat(cell);

  let mut elmat = na::DMatrix::zeros(kwhitney_basis_size, kwhitney_basis_size);
  let simplicies: Vec<_> = IndexSubsets::canonical(n + 1, difk).collect();
  let forms: Vec<Vec<_>> = simplicies
    .iter()
    .map(|simp| (0..difk).map(|i| assemble_const_form(simp, i, n)).collect())
    .collect();

  for (arank, asimp) in simplicies.iter().enumerate() {
    for (brank, bsimp) in simplicies.iter().enumerate() {
      let mut sum = 0.0;

      for l in 0..difk {
        for m in 0..difk {
          let sign = Sign::from_parity(l + m);

          let aform = &forms[arank][l];
          let bform = &forms[brank][m];

          let inner = cell.metric().kform_inner_product(k, aform, bform);
          sum += sign.as_f64() * inner * scalar_mass[(asimp[l], bsimp[m])];
        }
      }

      elmat[(arank, brank)] = k_factorial_sqr * sum;
    }
  }

  elmat
}

fn assemble_const_form(
  simp: &IndexAlgebra<Local, Sorted, Unsigned>,
  ignored_ivertex: usize,
  n: Dim,
) -> na::DVector<f64> {
  let k = simp.len() - 1;
  let kform_basis_size = binomial(n, k);

  let mut form = na::DVector::zeros(kform_basis_size);

  let mut form_indices = Vec::new();
  for (ivertex, &vertex) in simp.iter().enumerate() {
    if vertex != 0 && ivertex != ignored_ivertex {
      form_indices.push(vertex - 1);
    }
  }

  if simp[0] == 0 && ignored_ivertex != 0 {
    for i in 0..n {
      let mut form_indices = form_indices.clone();
      form_indices.insert(0, i);
      let Some(form_indices) = IndexAlgebra::new(form_indices).try_sort_signed() else {
        continue;
      };
      let sort_sign = form_indices.sign();
      let form_indices = form_indices.forget_sign().with_local_base(n);
      form[form_indices.lex_rank()] += -1.0 * sort_sign.as_f64();
    }
  } else {
    let form_indices = IndexAlgebra::new(form_indices.clone())
      .assume_sorted()
      .with_local_base(n);
    form[form_indices.lex_rank()] += 1.0;
  }

  form
}

/// Exact Element Matrix Provider for the $(dif u, dif v)$.
///
/// $A = [inner(dif lambda_J, dif lambda_I)_(L^2 Lambda^(k+1) (K))]_(I,J in Delta_k (K))$
pub fn dif_dif_elmat(cell: &LocalComplex, k: ExteriorRank) -> na::DMatrix<f64> {
  exterior_derivative(cell.dim(), k).transpose()
    * hodge_mass_elmat(cell, k + 1)
    * exterior_derivative(cell.dim(), k)
}

/// Exact Element Matrix Provider for the $(u, dif tau)$.
///
/// $A = [inner(lambda_J, dif lambda_I)_(L^2 Lambda^k (K))]_(I in Delta_(k-1), J in Delta_k (K))$
pub fn id_dif_elmat(cell: &LocalComplex, k: ExteriorRank) -> na::DMatrix<f64> {
  exterior_derivative(cell.dim(), k - 1).transpose() * hodge_mass_elmat(cell, k)
}

/// Exact Element Matrix Provider for the $(dif sigma, v)$.
///
/// $A = [inner(dif lambda_J, lambda_I)_(L^2 Lambda^k (K))]_(I in Delta_, J in Delta_(k-1) (K))$
pub fn dif_id_elmat(cell: &LocalComplex, k: ExteriorRank) -> na::DMatrix<f64> {
  hodge_mass_elmat(cell, k) * exterior_derivative(cell.dim(), k - 1)
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
      let kform_comb: Vec<_> = whitney_comb.iter().skip(1).map(|c| *c - 1).collect();
      for i in 0..n {
        let mut kform_comb = kform_comb.clone();
        kform_comb.insert(0, i);
        let sign = sort_signed(&mut kform_comb);

        kform_comb.dedup();
        if kform_comb.len() != difk {
          continue;
        };

        let kform_comb = IndexAlgebra::from(kform_comb)
          .assume_sorted()
          .with_local_base(n);
        let kform_comb_rank = kform_comb.lex_rank();
        ref_difwhitneys[(kform_comb_rank, whitney_comb_rank)] = -sign.as_f64() * difk_factorial;
      }
    } else {
      let kform_comb: Vec<_> = whitney_comb.iter().map(|c| *c - 1).collect();
      let kform_comb = IndexAlgebra::from(kform_comb)
        .assume_sorted()
        .with_local_base(n);
      let kform_comb_rank = kform_comb.lex_rank();
      ref_difwhitneys[(kform_comb_rank, whitney_comb_rank)] = difk_factorial;
    }
  }
  ref_difwhitneys
}

#[cfg(test)]
mod test {
  use crate::fe::{
    laplace_beltrami_elmat, ref_difbarys, scalar_mass_elmat, whitney::dif_dif_elmat,
  };

  use super::{hodge_mass_elmat, ref_difwhitneys};
  use common::linalg::assert_mat_eq;
  use exterior::RiemannianMetricExt;
  use manifold::simplicial::ReferenceCell;

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
      let whitneys = ref_difwhitneys(n, n);
      let zero = na::DMatrix::zeros(0, 1);
      assert_mat_eq(&whitneys, &zero)
    }
  }

  #[test]
  fn dif_dif0_is_laplace_beltrami() {
    for n in 1..=3 {
      let cell = ReferenceCell::new(n).to_cell_complex();
      let hodge_laplace = dif_dif_elmat(&cell, 0);
      let laplace_beltrami = laplace_beltrami_elmat(&cell);
      assert_mat_eq(&hodge_laplace, &laplace_beltrami);
    }
  }

  #[test]
  fn hodge_mass0_is_scalar_mass() {
    for n in 0..=3 {
      let cell = ReferenceCell::new(n).to_cell_complex();
      let hodge_mass = hodge_mass_elmat(&cell, 0);
      let scalar_mass = scalar_mass_elmat(&cell);
      assert_mat_eq(&hodge_mass, &scalar_mass);
    }
  }

  #[test]
  fn dif_dif_is_norm_of_difwhitneys() {
    for n in 0..=3 {
      let cell = ReferenceCell::new(n).to_cell_complex();
      for k in 0..n {
        let var0 = dif_dif_elmat(&cell, k);

        let var1 = cell.vol()
          * cell
            .metric()
            .kform_norm_sqr(k + 1, &ref_difwhitneys(cell.dim(), k));

        assert_mat_eq(&var0, &var1);
      }
    }
  }
}
