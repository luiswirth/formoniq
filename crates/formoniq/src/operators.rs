use common::sparse::SparseMatrix;
use exterior::{ExteriorRank, RiemannianMetricExt};
use geometry::metric::manifold::{local::LocalMetricComplex, MetricComplex};
use index_algebra::{binomial, combinators::IndexSubsets, factorial, sign::Sign, IndexSet};
use topology::{
  complex::{attribute::Cochain, local::LocalComplex, ManifoldComplex},
  simplex::SortedSimplex,
  Dim,
};

pub type DofIdx = usize;
pub type DofCoeff = f64;

// TODO: turn into cochain
pub type FeFunction = Cochain<Dim>;

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

pub trait ManifoldComplexExt {
  fn exterior_derivative_operator(&self, rank: ExteriorRank) -> SparseMatrix;
}
impl ManifoldComplexExt for ManifoldComplex {
  /// $dif^k: cal(W) Lambda^k -> cal(W) Lambda^(k+1)$
  fn exterior_derivative_operator(&self, rank: ExteriorRank) -> SparseMatrix {
    self.boundary_operator(rank + 1).transpose()
  }
}

pub trait LocalComplexExt {
  fn exterior_derivative_operator(&self, rank: ExteriorRank) -> na::DMatrix<f64>;
}
impl LocalComplexExt for LocalComplex {
  fn exterior_derivative_operator(&self, rank: ExteriorRank) -> na::DMatrix<f64> {
    self.boundary_operator(rank + 1).transpose()
  }
}

/// Element Matrix for the weak Hodge star operator / the mass bilinear form.
///
/// $M = [inner(star lambda_tau, lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$
pub struct HodgeMassElmat(pub ExteriorRank);
impl ElMatProvider for HodgeMassElmat {
  fn row_rank(&self) -> ExteriorRank {
    self.0
  }
  fn col_rank(&self) -> ExteriorRank {
    self.0
  }
  fn eval(&self, local_complex: &LocalMetricComplex) -> na::DMatrix<f64> {
    let n = local_complex.dim();
    let k = self.0;
    let difk = k + 1;
    let k_factorial = factorial(k) as f64;
    let k_factorial_sqr = k_factorial.powi(2);

    let kwhitney_basis_size = binomial(n + 1, k + 1);

    let scalar_mass = ScalarMassElmat.eval(local_complex);

    let mut elmat = na::DMatrix::zeros(kwhitney_basis_size, kwhitney_basis_size);
    let simplicies: Vec<_> = IndexSubsets::canonical(n + 1, difk).collect();
    let forms: Vec<Vec<_>> = simplicies
      .iter()
      .map(|simp| {
        (0..difk)
          .map(|i| construct_const_form(simp, i, n))
          .collect()
      })
      .collect();

    for (arank, asimp) in simplicies.iter().enumerate() {
      for (brank, bsimp) in simplicies.iter().enumerate() {
        let mut sum = 0.0;

        for l in 0..difk {
          for m in 0..difk {
            let sign = Sign::from_parity(l + m);

            let aform = &forms[arank][l];
            let bform = &forms[brank][m];

            let inner = local_complex.metric().kform_inner_product(k, aform, bform);
            sum += sign.as_f64() * inner * scalar_mass[(asimp[l], bsimp[m])];
          }
        }

        elmat[(arank, brank)] = k_factorial_sqr * sum;
      }
    }

    elmat
  }
}

fn construct_const_form(
  simplex: &SortedSimplex,
  ignored_ivertex: usize,
  n: Dim,
) -> na::DVector<f64> {
  let k = simplex.len() - 1;
  let kform_basis_size = binomial(n, k);

  let mut form = na::DVector::zeros(kform_basis_size);

  let mut form_indices = Vec::new();
  for (ivertex, vertex) in simplex.iter().enumerate() {
    if vertex != 0 && ivertex != ignored_ivertex {
      form_indices.push(vertex - 1);
    }
  }

  if simplex[0] == 0 && ignored_ivertex != 0 {
    for i in 0..n {
      let mut form_indices = form_indices.clone();
      form_indices.insert(0, i);
      let Some(form_indices) = IndexSet::new(form_indices).try_into_sorted_signed() else {
        continue;
      };
      let sort_sign = form_indices.sign;
      let form_indices = form_indices.set;
      form[form_indices.lex_rank(n)] += -1.0 * sort_sign.as_f64();
    }
  } else {
    let form_indices = IndexSet::new(form_indices.clone()).assume_sorted();
    form[form_indices.lex_rank(n)] += 1.0;
  }

  form
}

/// Element Matrix Provider for the $(dif u, dif v)$ bilinear form.
///
/// $A = [inner(dif lambda_J, dif lambda_I)_(L^2 Lambda^(k+1) (K))]_(I,J in Delta_k (K))$
pub struct CodifDifElmat(pub ExteriorRank);
impl ElMatProvider for CodifDifElmat {
  fn row_rank(&self) -> ExteriorRank {
    self.0
  }
  fn col_rank(&self) -> ExteriorRank {
    self.0
  }
  fn eval(&self, local_complex: &LocalMetricComplex) -> na::DMatrix<f64> {
    let k = self.0;
    let dif = local_complex.topology().exterior_derivative_operator(k);
    let codif = dif.transpose();
    let mass = HodgeMassElmat(k + 1).eval(local_complex);
    codif * mass * dif
  }
}

/// Element Matrix Provider for the weak mixed exterior derivative $(dif sigma, v)$.
///
/// $A = [inner(dif lambda_J, lambda_I)_(L^2 Lambda^k (K))]_(I in Delta_, J in Delta_(k-1) (K))$
pub struct DifElmat(pub ExteriorRank);
impl ElMatProvider for DifElmat {
  fn row_rank(&self) -> ExteriorRank {
    self.0
  }
  fn col_rank(&self) -> ExteriorRank {
    self.0 - 1
  }
  fn eval(&self, local_complex: &LocalMetricComplex) -> na::DMatrix<f64> {
    let k = self.0;
    let dif = local_complex.topology().exterior_derivative_operator(k - 1);
    let mass = HodgeMassElmat(k).eval(local_complex);
    mass * dif
  }
}

/// Element Matrix Provider for the weak mixed codifferential $(u, dif tau)$.
///
/// $A = [inner(lambda_J, dif lambda_I)_(L^2 Lambda^k (K))]_(I in Delta_(k-1), J in Delta_k (K))$
pub struct CodifElmat(pub ExteriorRank);
impl ElMatProvider for CodifElmat {
  fn row_rank(&self) -> ExteriorRank {
    self.0 - 1
  }
  fn col_rank(&self) -> ExteriorRank {
    self.0
  }
  fn eval(&self, local_complex: &LocalMetricComplex) -> na::DMatrix<f64> {
    let k = self.0;
    let dif = local_complex.topology().exterior_derivative_operator(k - 1);
    let codif = dif.transpose();
    let mass = HodgeMassElmat(k).eval(local_complex);
    codif * mass
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
      sum += func[vertex.to_dyn()];
    }
    let nvertices = facet.nvertices();
    let vol = mesh.local_complex(facet).vol();
    norm += (vol / nvertices as f64) * sum;
  }
  norm.sqrt()
}

#[cfg(test)]
mod test {
  use super::{CodifDifElmat, CodifElmat, DifElmat, HodgeMassElmat};
  use crate::{
    operators::{ref_difbarys, ElMatProvider, LaplaceBeltramiElmat, ScalarMassElmat},
    whitney::ref_difwhitneys,
  };

  use common::linalg::assert_mat_eq;
  use exterior::RiemannianMetricExt;
  use geometry::metric::manifold::local::LocalMetricComplex;

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
      let complex = LocalMetricComplex::reference(n);
      let hodge_laplace = CodifDifElmat(0).eval(&complex);
      let laplace_beltrami = LaplaceBeltramiElmat.eval(&complex);
      assert_mat_eq(&hodge_laplace, &laplace_beltrami);
    }
  }

  #[test]
  fn hodge_mass0_is_scalar_mass() {
    for n in 0..=3 {
      let complex = LocalMetricComplex::reference(n);
      let hodge_mass = HodgeMassElmat(0).eval(&complex);
      let scalar_mass = ScalarMassElmat.eval(&complex);
      assert_mat_eq(&hodge_mass, &scalar_mass);
    }
  }

  #[test]
  fn hodge_mass_n2_k1() {
    let complex = LocalMetricComplex::reference(2);
    let computed = HodgeMassElmat(1).eval(&complex);
    let expected = na::dmatrix![
      1./3.,1./6.,0.   ;
      1./6.,1./3.,0.   ;
      0.   ,0.   ,1./6.;
    ];
    assert_mat_eq(&computed, &expected);
  }

  #[test]
  fn dif_n2_k1() {
    let complex = LocalMetricComplex::reference(2);
    let computed = DifElmat(1).eval(&complex);
    let expected = na::dmatrix![
      -1./2., 1./3.,1./6.;
      -1./2., 1./6.,1./3.;
       0.   ,-1./6.,1./6.;
    ];
    assert_mat_eq(&computed, &expected);
  }

  #[test]
  fn codif_n2_k1() {
    let complex = LocalMetricComplex::reference(2);
    let computed = CodifElmat(1).eval(&complex);
    let expected = na::dmatrix![
      -1./2., -1./2., 0.   ;
       1./3.,  1./6.,-1./6.;
       1./6.,  1./3., 1./6.;
    ];
    assert_mat_eq(&computed, &expected);
  }

  #[test]
  fn dif_dif_is_norm_of_difwhitneys() {
    for n in 0..=3 {
      let complex = LocalMetricComplex::reference(n);
      for k in 0..n {
        let var0 = CodifDifElmat(k).eval(&complex);

        let var1 = complex.vol()
          * complex
            .metric()
            .kform_norm_sqr(k + 1, &ref_difwhitneys(complex.dim(), k));

        assert_mat_eq(&var0, &var1);
      }
    }
  }
}
