use crate::{
  combo::{binomial, combinators::IndexSubsets, factorial, sort_signed, IndexSet},
  exterior::ExteriorRank,
  mesh::KSimplexIdx,
  simplicial::REFCELLS,
  Dim,
};

pub type DofIdx = KSimplexIdx;

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
      let kform_comb: Vec<_> = whitney_comb.iter().skip(1).map(|c| *c - 1).collect();
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
      let kform_comb: Vec<_> = whitney_comb.iter().map(|c| *c - 1).collect();
      let kform_comb = IndexSet::from(kform_comb)
        .assume_sorted()
        .with_local_base(n);
      let kform_comb_rank = kform_comb.lex_rank();
      ref_difwhitneys[(kform_comb_rank, whitney_comb_rank)] = difk_factorial;
    }
  }
  ref_difwhitneys
}

/// The constant codifferentials of the reference Whitney forms, given in
/// the k-form standard basis.
pub fn ref_codifwhitneys(_n: Dim, _k: ExteriorRank) -> na::DMatrix<f64> {
  //let codif = (-1f64).powi((n * (k + 1) + 1) as i32)
  //  * khodge_star_local(cell, n - k + 1)
  //  * kexterior_derivative_local(cell.dim(), n - k)
  //  * khodge_star_local(cell, k);

  todo!("can we even compute this exactly?")
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

/// $dif^k: cal(W) Lambda^k -> cal(W) Lambda^(k+1)$
pub fn kexterior_derivative_local(cell_dim: Dim, k: ExteriorRank) -> na::DMatrix<f64> {
  REFCELLS[cell_dim].kboundary_operator(k + 1).transpose()
}

#[cfg(test)]
mod test {
  use super::{ref_difbarys, ref_difwhitneys};
  use crate::linalg::assert_mat_eq;

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
}
