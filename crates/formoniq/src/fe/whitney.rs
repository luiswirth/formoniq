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

/// Exact Element Matrix Provider for the exterior derivative part of Hodge-Laplace operator.
///
/// $A = [inner(dif lambda_tau, dif lambda_sigma)_(L^2 Lambda^(k+1) (K))]_(sigma,tau in Delta_k (K))$
pub fn hodge_laplace_dif_elmat(cell: &LocalComplex, k: ExteriorRank) -> na::DMatrix<f64> {
  let ref_difwhitneys = ref_difwhitneys(cell.dim(), k);
  println!("n={},k={k}{ref_difwhitneys}", cell.dim());
  cell.vol() * cell.metric().kform_norm_sqr(k + 1, &ref_difwhitneys)
}

/// $dif^k: cal(W) Lambda^k -> cal(W) Lambda^(k+1)$
pub fn kexterior_derivative_local(cell_dim: Dim, k: ExteriorRank) -> na::DMatrix<f64> {
  REFCELLS[cell_dim].kboundary_operator(k + 1).transpose()
}

#[cfg(test)]
mod test {
  use crate::fe::{laplace_beltrami_elmat, scalar_mass_elmat};

  use super::{
    hodge_laplace_dif_elmat, hodge_mass_elmat, kexterior_derivative_local, ref_difbarys,
    ref_difwhitneys,
  };
  use common::linalg::assert_mat_eq;
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
      println!("n={n}");
      let whitneys = ref_difwhitneys(n, n);
      let zero = na::DMatrix::zeros(0, 1);
      assert_mat_eq(&whitneys, &zero)
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
  fn hodge_laplace_dif_with_hodge_mass() {
    for n in 0..=3 {
      let cell = ReferenceCell::new(n).to_cell_complex();
      for k in 0..n {
        println!("n={n},k={k}");
        let var0 = kexterior_derivative_local(cell.dim(), k).transpose()
          * hodge_mass_elmat(&cell, k + 1)
          * kexterior_derivative_local(cell.dim(), k);
        let var1 = hodge_laplace_dif_elmat(&cell, k);
        assert_mat_eq(&var0, &var1);
      }
    }
  }
}
