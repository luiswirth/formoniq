use {
  common::{
    combo::{factorial, Sign},
    linalg::nalgebra::{Matrix, Vector},
  },
  ddf::{whitney::WhitneyLsf, ManifoldComplexExt},
  exterior::{list::ExteriorElementList, term::multi_gramian, Dim, ExteriorGrade},
  manifold::{
    geometry::{coord::simplex::SimplexCoords, metric::simplex::SimplexLengths},
    topology::{
      complex::Complex,
      simplex::{standard_subsimps, Simplex},
    },
  },
};

pub type DofIdx = usize;
pub type DofCoeff = f64;

pub type ElMat = Matrix;
pub trait ElMatProvider: Sync {
  fn row_grade(&self) -> ExteriorGrade;
  fn col_grade(&self) -> ExteriorGrade;
  fn eval(&self, geometry: &SimplexLengths) -> ElMat;
}

pub type ElVec = Vector;
pub trait ElVecProvider: Sync {
  fn grade(&self) -> ExteriorGrade;
  fn eval(&self, geometry: &SimplexLengths) -> ElVec;
}

/// Exact Element Matrix Provider for the Laplace-Beltrami operator.
///
/// $A = [(dif lambda_tau, dif lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$
pub struct LaplaceBeltramiElmat {
  dim: Dim,
  ref_difbarys: Matrix,
}
impl LaplaceBeltramiElmat {
  pub fn new(dim: Dim) -> Self {
    let ref_difbarys = SimplexCoords::standard(dim).difbarys().transpose();
    Self { dim, ref_difbarys }
  }
}
impl ElMatProvider for LaplaceBeltramiElmat {
  fn row_grade(&self) -> ExteriorGrade {
    0
  }
  fn col_grade(&self) -> ExteriorGrade {
    0
  }
  fn eval(&self, geometry: &SimplexLengths) -> ElMat {
    assert!(self.dim == geometry.dim());
    geometry.vol()
      * geometry
        .to_regge_metric()
        .inverse()
        .norm_sq_mat(&self.ref_difbarys)
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
  fn eval(&self, geometry: &SimplexLengths) -> ElMat {
    let ndofs = geometry.nvertices();
    let dim = geometry.dim();
    let v = geometry.vol() / ((dim + 1) * (dim + 2)) as f64;
    let mut elmat = Matrix::from_element(ndofs, ndofs, v);
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
  fn eval(&self, geomery: &SimplexLengths) -> ElMat {
    let n = geomery.nvertices();
    let v = geomery.vol() / n as f64;
    Matrix::from_diagonal_element(n, n, v)
  }
}

/// Element Matrix for the weak Hodge star operator / the mass bilinear form.
///
/// $M = [inner(star lambda_tau, lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$
pub struct HodgeMassElmat {
  dim: Dim,
  grade: ExteriorGrade,
  simplicies: Vec<Simplex>,
  wedge_terms: Vec<ExteriorElementList>,
}
impl HodgeMassElmat {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    let simplicies: Vec<_> = standard_subsimps(dim, grade).collect();
    let wedge_terms: Vec<ExteriorElementList> = simplicies
      .iter()
      .cloned()
      .map(|simp| WhitneyLsf::standard(dim, simp).wedge_terms().collect())
      .collect();

    Self {
      dim,
      grade,
      simplicies,
      wedge_terms,
    }
  }
}
impl ElMatProvider for HodgeMassElmat {
  fn row_grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.grade
  }

  fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    assert_eq!(self.dim, geometry.dim());

    let scalar_mass = ScalarMassElmat.eval(geometry);

    let mut elmat = Matrix::zeros(self.simplicies.len(), self.simplicies.len());
    for (i, asimp) in self.simplicies.iter().enumerate() {
      for (j, bsimp) in self.simplicies.iter().enumerate() {
        let wedge_terms_a = &self.wedge_terms[i];
        let wedge_terms_b = &self.wedge_terms[j];
        let wedge_inners = multi_gramian(&geometry.to_regge_metric().inverse(), self.grade)
          .inner_mat(wedge_terms_a.coeffs(), wedge_terms_b.coeffs());

        let nvertices = self.grade + 1;
        let mut sum = 0.0;
        for avertex in 0..nvertices {
          for bvertex in 0..nvertices {
            let sign = Sign::from_parity(avertex + bvertex);

            let inner = wedge_inners[(avertex, bvertex)];

            sum += sign.as_f64() * inner * scalar_mass[(asimp[avertex], bsimp[bvertex])];
          }
        }

        elmat[(i, j)] = sum;
      }
    }

    factorial(self.grade).pow(2) as f64 * elmat
  }
}

/// Element Matrix Provider for the weak mixed exterior derivative $(dif sigma, v)$.
///
/// $A = [inner(dif lambda_J, lambda_I)_(L^2 Lambda^k (K))]_(I in Delta_, J in Delta_(k-1) (K))$
pub struct DifElmat {
  mass: HodgeMassElmat,
  dif: Matrix,
}
impl DifElmat {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    let mass = HodgeMassElmat::new(dim, grade);
    let dif = Complex::standard(dim).exterior_derivative_operator(grade - 1);
    let dif = Matrix::from(&dif);
    Self { mass, dif }
  }
}

impl ElMatProvider for DifElmat {
  fn row_grade(&self) -> ExteriorGrade {
    self.mass.grade
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.mass.grade - 1
  }
  fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    let mass = self.mass.eval(geometry);
    mass * &self.dif
  }
}

/// Element Matrix Provider for the weak mixed codifferential $(u, dif tau)$.
///
/// $A = [inner(lambda_J, dif lambda_I)_(L^2 Lambda^k (K))]_(I in Delta_(k-1), J in Delta_k (K))$
pub struct CodifElmat {
  mass: HodgeMassElmat,
  codif: Matrix,
}
impl CodifElmat {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    let mass = HodgeMassElmat::new(dim, grade);
    let dif = Complex::standard(dim).exterior_derivative_operator(grade - 1);
    let dif = Matrix::from(&dif);
    let codif = dif.transpose();
    Self { mass, codif }
  }
}
impl ElMatProvider for CodifElmat {
  fn row_grade(&self) -> ExteriorGrade {
    self.mass.grade - 1
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.mass.grade
  }
  fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    let mass = self.mass.eval(geometry);
    &self.codif * mass
  }
}

/// Element Matrix Provider for the $(dif u, dif v)$ bilinear form.
///
/// $A = [inner(dif lambda_J, dif lambda_I)_(L^2 Lambda^(k+1) (K))]_(I,J in Delta_k (K))$
pub struct CodifDifElmat {
  mass: HodgeMassElmat,
  dif: Matrix,
  codif: Matrix,
}
impl CodifDifElmat {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    let mass = HodgeMassElmat::new(dim, grade + 1);
    let dif = Complex::standard(dim).exterior_derivative_operator(grade);
    let dif = Matrix::from(&dif);
    let codif = dif.transpose();

    Self { mass, dif, codif }
  }
}

impl ElMatProvider for CodifDifElmat {
  fn row_grade(&self) -> ExteriorGrade {
    self.mass.grade - 1
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.mass.grade - 1
  }
  fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    let mass = self.mass.eval(geometry);
    &self.codif * mass * &self.dif
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::operators::{ElMatProvider, LaplaceBeltramiElmat, ScalarMassElmat};

  use ddf::whitney::WhitneyLsf;
  use exterior::term::multi_gramian;
  use manifold::{geometry::metric::simplex::SimplexLengths, topology::simplex::standard_subsimps};

  use approx::assert_relative_eq;

  #[test]
  fn codifdif0_is_laplace_beltrami() {
    let grade = 0;
    for dim in 1..=3 {
      let geo = SimplexLengths::standard(dim);
      let hodge_laplace = CodifDifElmat::new(dim, grade).eval(&geo);
      let laplace_beltrami = LaplaceBeltramiElmat::new(dim).eval(&geo);
      assert_relative_eq!(&hodge_laplace, &laplace_beltrami);
    }
  }

  #[test]
  fn hodge_mass0_is_scalar_mass() {
    let grade = 0;
    for dim in 0..=3 {
      let geo = SimplexLengths::standard(dim);
      let hodge_mass = HodgeMassElmat::new(dim, grade).eval(&geo);
      let scalar_mass = ScalarMassElmat.eval(&geo);
      assert_relative_eq!(&hodge_mass, &scalar_mass);
    }
  }

  #[test]
  fn hodge_mass_dim2_grade1() {
    let dim = 2;
    let grade = 1;
    let geo = SimplexLengths::standard(dim);
    let computed = HodgeMassElmat::new(dim, grade).eval(&geo);
    let expected = na::dmatrix![
      1./3.,1./6.,0.   ;
      1./6.,1./3.,0.   ;
      0.   ,0.   ,1./6.;
    ];
    assert_relative_eq!(&computed, &expected);
  }

  #[test]
  fn dif_n2_k1() {
    let dim = 2;
    let grade = 1;
    let geo = SimplexLengths::standard(dim);
    let computed = DifElmat::new(dim, grade).eval(&geo);
    let expected = na::dmatrix![
      -1./2., 1./3.,1./6.;
      -1./2., 1./6.,1./3.;
       0.   ,-1./6.,1./6.;
    ];
    assert_relative_eq!(&computed, &expected);
  }

  #[test]
  fn codif_n2_k1() {
    let dim = 2;
    let grade = 1;
    let geo = SimplexLengths::standard(dim);
    let computed = CodifElmat::new(dim, grade).eval(&geo);
    let expected = na::dmatrix![
      -1./2., -1./2., 0.   ;
       1./3.,  1./6.,-1./6.;
       1./6.,  1./3., 1./6.;
    ];
    assert_relative_eq!(&computed, &expected);
  }

  #[test]
  fn dif_dif_is_norm_of_difwhitneys() {
    for dim in 1..=3 {
      let geo = SimplexLengths::standard(dim);
      for grade in 0..dim {
        let difdif = CodifDifElmat::new(dim, grade).eval(&geo);

        let difwhitneys: Vec<_> = standard_subsimps(dim, grade)
          .map(|simp| WhitneyLsf::standard(dim, simp).dif())
          .collect();
        let mut inner = Matrix::zeros(difwhitneys.len(), difwhitneys.len());
        for (i, awhitney) in difwhitneys.iter().enumerate() {
          for (j, bwhitney) in difwhitneys.iter().enumerate() {
            inner[(i, j)] = multi_gramian(&geo.to_regge_metric().inverse(), grade + 1)
              .inner(awhitney.coeffs(), bwhitney.coeffs());
          }
        }
        inner *= geo.vol();
        assert_relative_eq!(&difdif, &inner);
      }
    }
  }
}
