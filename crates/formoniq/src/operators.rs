use {
  derham::{interpolate::form::WhitneyForm, section::Section},
  exterior::{exterior_power, multiform_gramian, Covariant, Dim, ExteriorGrade},
  formoniq_linalg::nalgebra::{Matrix, Vector},
  gramian::RiemannianMetric,
  multiindex::{factorial, Combination},
  simplicial::{
    atlas::{ref_difbarys, MeshPoint, SimplexQuadRule},
    geometry::cell_volume,
    topology::{
      handle::SimplexRef,
      simplex::{standard_boundary_operator, standard_subsimps},
    },
  },
};

pub type ElMat = Matrix;
pub trait ElMatProvider: Sync {
  fn row_grade(&self) -> ExteriorGrade;
  fn col_grade(&self) -> ExteriorGrade;
  fn eval(&self, metric: &RiemannianMetric) -> ElMat;
}

/// Element matrix of the scalar mass bilinear form, $[integral_K lambda_i lambda_j]$.
///
/// Exact closed form: $vol(K) (1 + delta_(i j)) / ((n+1)(n+2))$.
/// The barycentric building block of the Hodge mass matrix.
fn scalar_mass_elmat(metric: &RiemannianMetric) -> ElMat {
  let dim = metric.dim();
  let ndofs = dim + 1;
  let v = cell_volume(metric) / ((dim + 1) * (dim + 2)) as f64;
  let mut elmat = Matrix::from_element(ndofs, ndofs, v);
  elmat.fill_diagonal(2.0 * v);
  elmat
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
  fn eval(&self, metric: &RiemannianMetric) -> ElMat {
    let n = metric.dim() + 1;
    let v = cell_volume(metric) / n as f64;
    Matrix::from_diagonal_element(n, n, v)
  }
}

/// Element Matrix for the weak Hodge star operator / the mass bilinear form.
///
/// $M = [inner(star lambda_tau, lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$
pub struct HodgeMassElmat {
  dim: Dim,
  grade: ExteriorGrade,
  simplices: Vec<Combination>,
  /// $Lambda^k$ of the reference barycentric differentials: the pullback
  /// matrix taking formal barycentric $k$-blades to reference $k$-forms.
  difbarys_power: Matrix,
}
impl HodgeMassElmat {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    let simplices: Vec<_> = standard_subsimps(dim, grade).collect();
    let difbarys_power = exterior_power(&ref_difbarys(dim), grade);

    Self {
      dim,
      grade,
      simplices,
      difbarys_power,
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

  fn eval(&self, metric: &RiemannianMetric) -> Matrix {
    assert_eq!(self.dim, metric.dim());

    let scalar_mass = scalar_mass_elmat(metric);
    let form_gramian = multiform_gramian(metric, self.grade);

    // Inner products of the pulled-back barycentric k-blades
    // $lambda^* (e_I)$: one Cauchy-Binet sandwich for all Whitney wedge
    // terms at once.
    let blade_inners =
      &self.difbarys_power * form_gramian.matrix() * self.difbarys_power.transpose();

    let mut elmat = Matrix::zeros(self.simplices.len(), self.simplices.len());
    for (i, asimp) in self.simplices.iter().enumerate() {
      for (j, bsimp) in self.simplices.iter().enumerate() {
        let mut sum = 0.0;
        for (asign, avertex, arest) in asimp.deletions() {
          for (bsign, bvertex, brest) in bsimp.deletions() {
            sum += (asign * bsign).as_f64()
              * blade_inners[(arest.rank(), brest.rank())]
              * scalar_mass[(avertex, bvertex)];
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
    let dif = standard_boundary_operator(dim, grade).transpose();
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
  fn eval(&self, metric: &RiemannianMetric) -> Matrix {
    let mass = self.mass.eval(metric);
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
    let codif = standard_boundary_operator(dim, grade);
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
  fn eval(&self, metric: &RiemannianMetric) -> Matrix {
    let mass = self.mass.eval(metric);
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
    let dif = standard_boundary_operator(dim, grade + 1).transpose();
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
  fn eval(&self, metric: &RiemannianMetric) -> Matrix {
    let mass = self.mass.eval(metric);
    &self.codif * mass * &self.dif
  }
}

pub type ElVec = Vector;
pub trait ElVecProvider: Sync {
  fn grade(&self) -> ExteriorGrade;
  fn eval(&self, metric: &RiemannianMetric, cell: SimplexRef) -> ElVec;
}

/// Element vector of the source load
/// $[integral_K inner(f, W_sigma)_(Lambda^k) vol]_(sigma in Delta_k (K))$.
///
/// Intrinsic: the source is a field on the manifold, the Whitney shape
/// functions are the reference ones, and both are paired in the cell's
/// reference frame under the induced inner product $Lambda^k g^(-1)$ of the
/// cell metric. Source assembly therefore runs on Regge geometry, with no
/// coordinates in sight.
pub struct SourceElVec<'a, F> {
  source: &'a F,
  qr: SimplexQuadRule,
  whitneys: Vec<WhitneyForm>,
}
impl<'a, F: Section<Covariant>> SourceElVec<'a, F> {
  pub fn new(source: &'a F, qr: Option<SimplexQuadRule>) -> Self {
    let dim = source.dim();
    let qr = qr.unwrap_or(SimplexQuadRule::degree(dim, 1));
    let whitneys = standard_subsimps(dim, source.grade())
      .map(|dof_simp| WhitneyForm::standard(dim, dof_simp))
      .collect();
    Self {
      source,
      qr,
      whitneys,
    }
  }
}
impl<F: Sync + Section<Covariant>> ElVecProvider for SourceElVec<'_, F> {
  fn grade(&self) -> ExteriorGrade {
    self.source.grade()
  }
  fn eval(&self, metric: &RiemannianMetric, cell: SimplexRef) -> ElVec {
    let inner = multiform_gramian(metric, self.grade());

    let mut elvec = ElVec::zeros(self.whitneys.len());
    for (iwhitney, whitney) in self.whitneys.iter().enumerate() {
      let inner_pointwise = |point: &MeshPoint| {
        inner.inner(
          self.source.at(point).coeffs(),
          whitney.at_bary(point.bary()).coeffs(),
        )
      };
      elvec[iwhitney] = self
        .qr
        .integrate_cell(cell.idx(), &inner_pointwise, cell_volume(metric));
    }
    elvec
  }
}

#[cfg(test)]
mod test {
  use super::*;

  use derham::interpolate::form::WhitneyForm;
  use simplicial::{
    geometry::metric::simplex::SimplexLengths, topology::simplex::standard_subsimps,
  };

  use approx::assert_relative_eq;

  #[test]
  fn hodge_mass0_is_scalar_mass() {
    for dim in 0..=3 {
      let geo = SimplexLengths::standard(dim);
      let hodge_mass = HodgeMassElmat::new(dim, 0).eval(&geo.riemannian_metric());
      let scalar_mass = scalar_mass_elmat(&geo.riemannian_metric());
      assert_relative_eq!(&hodge_mass, &scalar_mass);
    }
  }

  #[test]
  fn hodge_mass_dim2_grade1() {
    let dim = 2;
    let grade = 1;
    let geo = SimplexLengths::standard(dim);
    let computed = HodgeMassElmat::new(dim, grade).eval(&geo.riemannian_metric());
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
    let computed = DifElmat::new(dim, grade).eval(&geo.riemannian_metric());
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
    let computed = CodifElmat::new(dim, grade).eval(&geo.riemannian_metric());
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
        let difdif = CodifDifElmat::new(dim, grade).eval(&geo.riemannian_metric());

        let difwhitneys: Vec<_> = standard_subsimps(dim, grade)
          .map(|simp| WhitneyForm::standard(dim, simp).dif())
          .collect();
        let mut inner = Matrix::zeros(difwhitneys.len(), difwhitneys.len());
        for (i, awhitney) in difwhitneys.iter().enumerate() {
          for (j, bwhitney) in difwhitneys.iter().enumerate() {
            inner[(i, j)] = awhitney.inner(bwhitney, &geo.riemannian_metric());
          }
        }
        inner *= geo.vol();
        assert_relative_eq!(&difdif, &inner);
      }
    }
  }
}
