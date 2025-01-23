use common::sparse::SparseMatrix;
use exterior::{ExteriorGrade, RiemannianMetricExt};
use geometry::{
  coord::manifold::CoordSimplex,
  metric::manifold::{local::LocalMetricComplex, MetricComplex},
};
use index_algebra::{factorial, sign::Sign};
use topology::{
  complex::{attribute::Cochain, local::LocalComplex, TopologyComplex},
  simplex::subsimplicies,
  Dim,
};

use crate::whitney::WhitneyForm;

pub type DofIdx = usize;
pub type DofCoeff = f64;

// TODO: turn into cochain
pub type FeFunction = Cochain<Dim>;

pub type ElMat = na::DMatrix<f64>;
pub trait ElMatProvider {
  fn row_grade(&self) -> ExteriorGrade;
  fn col_grade(&self) -> ExteriorGrade;
  fn eval(&self, local_complex: &LocalMetricComplex) -> ElMat;
}

pub type ElVec = na::DVector<f64>;
pub trait ElVecProvider {
  fn grade(&self) -> ExteriorGrade;
  fn eval(&self, local_complex: &LocalMetricComplex) -> ElVec;
}

/// Exact Element Matrix Provider for the Laplace-Beltrami operator.
///
/// $A = [(dif lambda_tau, dif lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$
pub struct LaplaceBeltramiElmat;
impl ElMatProvider for LaplaceBeltramiElmat {
  fn row_grade(&self) -> ExteriorGrade {
    0
  }
  fn col_grade(&self) -> ExteriorGrade {
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
  fn row_grade(&self) -> ExteriorGrade {
    0
  }
  fn col_grade(&self) -> ExteriorGrade {
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
  fn row_grade(&self) -> ExteriorGrade {
    0
  }
  fn col_grade(&self) -> ExteriorGrade {
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
  fn grade(&self) -> ExteriorGrade {
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
  fn exterior_derivative_operator(&self, grade: ExteriorGrade) -> SparseMatrix;
}
impl ManifoldComplexExt for TopologyComplex {
  /// $dif^k: cal(W) Lambda^k -> cal(W) Lambda^(k+1)$
  fn exterior_derivative_operator(&self, grade: ExteriorGrade) -> SparseMatrix {
    self.boundary_operator(grade + 1).transpose()
  }
}

pub trait LocalComplexExt {
  fn exterior_derivative_operator(&self, grade: ExteriorGrade) -> na::DMatrix<f64>;
}
impl LocalComplexExt for LocalComplex {
  fn exterior_derivative_operator(&self, grade: ExteriorGrade) -> na::DMatrix<f64> {
    self.boundary_operator(grade + 1).transpose()
  }
}

/// Element Matrix for the weak Hodge star operator / the mass bilinear form.
///
/// $M = [inner(star lambda_tau, lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$
pub struct HodgeMassElmat(pub ExteriorGrade);
impl ElMatProvider for HodgeMassElmat {
  fn row_grade(&self) -> ExteriorGrade {
    self.0
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.0
  }

  fn eval(&self, local_complex: &LocalMetricComplex) -> na::DMatrix<f64> {
    let dim = local_complex.dim();
    let grade = self.0;

    let nvertices = grade + 1;
    let simplicies: Vec<_> = subsimplicies(dim, grade).collect();

    let wedge_terms: Vec<_> = simplicies
      .iter()
      .cloned()
      .map(|simp| WhitneyForm::new(CoordSimplex::standard(dim), simp).wedge_terms())
      .collect();

    let scalar_mass = ScalarMassElmat.eval(local_complex);

    let mut elmat = na::DMatrix::zeros(simplicies.len(), simplicies.len());
    for (i, asimp) in simplicies.iter().enumerate() {
      for (j, bsimp) in simplicies.iter().enumerate() {
        let wedge_terms_a = &wedge_terms[i];
        let wedge_terms_b = &wedge_terms[j];
        let wedge_inners = local_complex
          .metric()
          .multi_form_inner_product_mat(wedge_terms_a, wedge_terms_b);

        let mut sum = 0.0;
        for avertex in 0..nvertices {
          for bvertex in 0..nvertices {
            let sign = Sign::from_parity(avertex + bvertex);

            let inner = wedge_inners[(avertex, bvertex)];

            sum += sign.as_f64()
              * inner
              * scalar_mass[(asimp.vertices[avertex], bsimp.vertices[bvertex])];
          }
        }

        elmat[(i, j)] = sum;
      }
    }

    (factorial(grade) as f64).powi(2) * elmat
  }
}

/// Element Matrix Provider for the $(dif u, dif v)$ bilinear form.
///
/// $A = [inner(dif lambda_J, dif lambda_I)_(L^2 Lambda^(k+1) (K))]_(I,J in Delta_k (K))$
pub struct CodifDifElmat(pub ExteriorGrade);
impl ElMatProvider for CodifDifElmat {
  fn row_grade(&self) -> ExteriorGrade {
    self.0
  }
  fn col_grade(&self) -> ExteriorGrade {
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
pub struct DifElmat(pub ExteriorGrade);
impl ElMatProvider for DifElmat {
  fn row_grade(&self) -> ExteriorGrade {
    self.0
  }
  fn col_grade(&self) -> ExteriorGrade {
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
pub struct CodifElmat(pub ExteriorGrade);
impl ElMatProvider for CodifElmat {
  fn row_grade(&self) -> ExteriorGrade {
    self.0 - 1
  }
  fn col_grade(&self) -> ExteriorGrade {
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
    operators::{ElMatProvider, LaplaceBeltramiElmat, ScalarMassElmat},
    whitney::WhitneyForm,
  };

  use common::linalg::assert_mat_eq;
  use exterior::RiemannianMetricExt;
  use geometry::{
    coord::manifold::{CoordComplex, SimplexHandleExt},
    metric::manifold::local::LocalMetricComplex,
  };

  #[test]
  fn dif_dif0_is_laplace_beltrami() {
    for n in 1..=3 {
      let complex = LocalMetricComplex::reference(n);
      let hodge_laplace = CodifDifElmat(0).eval(&complex);
      let laplace_beltrami = LaplaceBeltramiElmat.eval(&complex);
      assert_mat_eq(&hodge_laplace, &laplace_beltrami, None);
    }
  }

  #[test]
  fn hodge_mass0_is_scalar_mass() {
    for n in 0..=3 {
      let complex = LocalMetricComplex::reference(n);
      let hodge_mass = HodgeMassElmat(0).eval(&complex);
      let scalar_mass = ScalarMassElmat.eval(&complex);
      assert_mat_eq(&hodge_mass, &scalar_mass, None);
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
    assert_mat_eq(&computed, &expected, None);
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
    assert_mat_eq(&computed, &expected, None);
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
    assert_mat_eq(&computed, &expected, None);
  }

  #[test]
  fn dif_dif_is_norm_of_difwhitneys() {
    for dim in 1..=3 {
      let coord_complex = CoordComplex::standard(dim);
      let metric_complex = coord_complex.to_metric_complex();
      for grade in 0..dim {
        let facet = metric_complex.topology().facets().get_by_kidx(0);
        let coord_facet = facet.coord_simplex(coord_complex.coords());
        let local_complex = metric_complex.local_complex(facet);

        let difdif = CodifDifElmat(grade).eval(&local_complex);

        let difwhitneys: Vec<_> = metric_complex
          .topology()
          .skeleton(grade)
          .iter()
          .map(|simp| WhitneyForm::new(coord_facet.clone(), simp.simplex_set().clone()).dif())
          .collect();
        let mut inner = na::DMatrix::zeros(difwhitneys.len(), difwhitneys.len());
        for (i, awhitney) in difwhitneys.iter().enumerate() {
          for (j, bwhitney) in difwhitneys.iter().enumerate() {
            inner[(i, j)] = local_complex
              .metric()
              .multi_form_inner_product(awhitney, bwhitney);
          }
        }
        inner *= local_complex.vol();
        assert_mat_eq(&difdif, &inner, None);
      }
    }
  }
}
