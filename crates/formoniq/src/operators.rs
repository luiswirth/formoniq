use {
  exterior::{term::RiemannianMetricExt, ExteriorGrade},
  manifold::{
    geometry::metric::SimplexGeometry,
    topology::{
      complex::{DIM_PRECOMPUTED, LOCAL_BOUNDARY_OPERATORS},
      simplex::{nsubsimplicies, standard_subsimps},
    },
    Dim,
  },
  multi_index::{factorial, sign::Sign},
  whitney::WhitneyRefLsf,
};

use std::sync::LazyLock;

pub static LOCAL_DIFFERENTIAL_OPERATORS: LazyLock<Vec<Vec<na::DMatrix<f64>>>> =
  LazyLock::new(|| {
    (0..=DIM_PRECOMPUTED)
      .map(|dim| {
        (0..=dim)
          .map(|rank| {
            if rank < dim {
              LOCAL_BOUNDARY_OPERATORS[dim][rank + 1].transpose()
            } else {
              na::DMatrix::zeros(0, nsubsimplicies(dim, rank))
            }
          })
          .collect()
      })
      .collect()
  });

pub type DofIdx = usize;
pub type DofCoeff = f64;

pub type ElMat = na::DMatrix<f64>;
pub trait ElMatProvider: Sync {
  fn row_grade(&self) -> ExteriorGrade;
  fn col_grade(&self) -> ExteriorGrade;
  fn eval(&self, geometry: &SimplexGeometry) -> ElMat;
}

pub type ElVec = na::DVector<f64>;
pub trait ElVecProvider: Sync {
  fn grade(&self) -> ExteriorGrade;
  fn eval(&self, geometry: &SimplexGeometry) -> ElVec;
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
  fn eval(&self, geometry: &SimplexGeometry) -> ElMat {
    let ref_difbarys = ref_difbarys(geometry.dim());
    geometry.vol() * geometry.metric().covector_norm_sqr(&ref_difbarys)
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
  fn eval(&self, geometry: &SimplexGeometry) -> ElMat {
    let ndofs = geometry.nvertices();
    let dim = geometry.dim();
    let v = geometry.vol() / ((dim + 1) * (dim + 2)) as f64;
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
  fn eval(&self, geomery: &SimplexGeometry) -> ElMat {
    let n = geomery.nvertices();
    let v = geomery.vol() / n as f64;
    na::DMatrix::from_diagonal_element(n, n, v)
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

  // TODO: store precomputed values
  fn eval(&self, geometry: &SimplexGeometry) -> na::DMatrix<f64> {
    let dim = geometry.dim();
    let grade = self.0;

    let scalar_mass = ScalarMassElmat.eval(geometry);

    let nvertices = grade + 1;
    let simplicies: Vec<_> = standard_subsimps(dim, grade).collect();

    let wedge_terms: Vec<_> = simplicies
      .iter()
      .cloned()
      .map(|simp| WhitneyRefLsf::new(dim, simp).wedge_terms().collect())
      .collect();

    let mut elmat = na::DMatrix::zeros(simplicies.len(), simplicies.len());
    for (i, asimp) in simplicies.iter().enumerate() {
      for (j, bsimp) in simplicies.iter().enumerate() {
        let wedge_terms_a = &wedge_terms[i];
        let wedge_terms_b = &wedge_terms[j];
        let wedge_inners = geometry
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
  fn eval(&self, geometry: &SimplexGeometry) -> na::DMatrix<f64> {
    let dim = geometry.dim();
    let grade = self.0;
    let dif = &LOCAL_DIFFERENTIAL_OPERATORS[dim][grade - 1];
    let mass = HodgeMassElmat(grade).eval(geometry);
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
  fn eval(&self, geometry: &SimplexGeometry) -> na::DMatrix<f64> {
    let dim = geometry.dim();
    let grade = self.0;
    let dif = &LOCAL_DIFFERENTIAL_OPERATORS[dim][grade - 1];
    let codif = dif.transpose();
    let mass = HodgeMassElmat(grade).eval(geometry);
    codif * mass
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
  fn eval(&self, geometry: &SimplexGeometry) -> na::DMatrix<f64> {
    let dim = geometry.dim();
    let grade = self.0;
    let dif = &LOCAL_DIFFERENTIAL_OPERATORS[dim][grade];
    let codif = dif.transpose();
    let mass = HodgeMassElmat(grade + 1).eval(geometry);
    codif * mass * dif
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

#[cfg(test)]
mod test {
  use super::{CodifDifElmat, CodifElmat, DifElmat, HodgeMassElmat};
  use crate::operators::{ElMatProvider, LaplaceBeltramiElmat, ScalarMassElmat};

  use common::linalg::assert_mat_eq;
  use exterior::term::RiemannianMetricExt;
  use manifold::{geometry::metric::SimplexGeometry, topology::simplex::standard_subsimps};
  use whitney::WhitneyRefLsf;

  #[test]
  fn dif_dif0_is_laplace_beltrami() {
    for n in 1..=3 {
      let complex = SimplexGeometry::standard(n);
      let hodge_laplace = CodifDifElmat(0).eval(&complex);
      let laplace_beltrami = LaplaceBeltramiElmat.eval(&complex);
      assert_mat_eq(&hodge_laplace, &laplace_beltrami, None);
    }
  }

  #[test]
  fn hodge_mass0_is_scalar_mass() {
    for n in 0..=3 {
      let complex = SimplexGeometry::standard(n);
      let hodge_mass = HodgeMassElmat(0).eval(&complex);
      let scalar_mass = ScalarMassElmat.eval(&complex);
      assert_mat_eq(&hodge_mass, &scalar_mass, None);
    }
  }

  #[test]
  fn hodge_mass_n2_k1() {
    let complex = SimplexGeometry::standard(2);
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
    let complex = SimplexGeometry::standard(2);
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
    let complex = SimplexGeometry::standard(2);
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
      let geometry = SimplexGeometry::standard(dim);
      for grade in 0..dim {
        let difdif = CodifDifElmat(grade).eval(&geometry);

        let difwhitneys: Vec<_> = standard_subsimps(dim, grade)
          .map(|simp| WhitneyRefLsf::new(dim, simp).dif())
          .collect();
        let mut inner = na::DMatrix::zeros(difwhitneys.len(), difwhitneys.len());
        for (i, awhitney) in difwhitneys.iter().enumerate() {
          for (j, bwhitney) in difwhitneys.iter().enumerate() {
            inner[(i, j)] = geometry
              .metric()
              .multi_form_inner_product(awhitney, bwhitney);
          }
        }
        inner *= geometry.vol();
        assert_mat_eq(&difdif, &inner, None);
      }
    }
  }
}
