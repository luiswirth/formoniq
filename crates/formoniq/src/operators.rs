use {
  common::{
    combo::{factorial, Combination},
    linalg::nalgebra::{Matrix, Vector},
  },
  ddf::whitney::lsf::WhitneyLsf,
  exterior::{exterior_power, field::ExteriorField, multi_gramian, Dim, ExteriorGrade},
  manifold::{
    geometry::{
      coord::{mesh::MeshCoords, quadrature::SimplexQuadRule, simplex::SimplexCoords, CoordRef},
      metric::simplex::SimplexLengths,
    },
    topology::simplex::{standard_boundary_operator, standard_subsimps, Simplex},
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

/// Element matrix of the scalar mass bilinear form, $[integral_K lambda_i lambda_j]$.
///
/// Exact closed form: $vol(K) (1 + delta_(i j)) / ((n+1)(n+2))$.
/// The barycentric building block of the Hodge mass matrix.
fn scalar_mass_elmat(geometry: &SimplexLengths) -> ElMat {
  let ndofs = geometry.nvertices();
  let dim = geometry.dim();
  let v = geometry.vol() / ((dim + 1) * (dim + 2)) as f64;
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
  simplices: Vec<Combination>,
  /// $Lambda^k$ of the reference barycentric differentials: the pullback
  /// matrix taking formal barycentric $k$-blades to reference $k$-forms.
  difbarys_power: Matrix,
}
impl HodgeMassElmat {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    let simplices: Vec<_> = standard_subsimps(dim, grade).collect();
    let ref_difbarys = SimplexCoords::standard(dim).difbarys();
    let difbarys_power = exterior_power(&ref_difbarys, grade);

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

  fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    assert_eq!(self.dim, geometry.dim());

    let scalar_mass = scalar_mass_elmat(geometry);
    let metric = geometry.riemannian_metric();
    let form_gramian = multi_gramian(metric.covector_gramian(), self.grade);

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
  fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    let mass = self.mass.eval(geometry);
    &self.codif * mass * &self.dif
  }
}

pub type ElVec = Vector;
pub trait ElVecProvider: Sync {
  fn grade(&self) -> ExteriorGrade;
  fn eval(&self, geometry: &SimplexLengths, topology: &Simplex) -> ElVec;
}

pub struct SourceElVec<'a, F>
where
  F: ExteriorField,
{
  source: &'a F,
  mesh_coords: &'a MeshCoords,
  qr: SimplexQuadRule,
  whitneys: Vec<WhitneyLsf>,
}
impl<'a, F> SourceElVec<'a, F>
where
  F: ExteriorField,
{
  pub fn new(source: &'a F, mesh_coords: &'a MeshCoords, qr: Option<SimplexQuadRule>) -> Self {
    let dim = source.dim_intrinsic();
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(dim));
    let whitneys = standard_subsimps(dim, source.grade())
      .map(|dof_simp| WhitneyLsf::standard(dim, dof_simp))
      .collect();
    Self {
      source,
      mesh_coords,
      qr,
      whitneys,
    }
  }
}
impl<F> ElVecProvider for SourceElVec<'_, F>
where
  F: Sync + ExteriorField,
{
  fn grade(&self) -> ExteriorGrade {
    self.source.grade()
  }
  fn eval(&self, geometry: &SimplexLengths, topology: &Simplex) -> ElVec {
    let cell_coords = SimplexCoords::from_simplex_and_coords(topology, self.mesh_coords);

    let inner = multi_gramian(geometry.riemannian_metric().covector_gramian(), self.grade());

    let mut elvec = ElVec::zeros(self.whitneys.len());
    for (iwhitney, whitney) in self.whitneys.iter().enumerate() {
      let inner_pointwise = |local: CoordRef| {
        let global = cell_coords.local2global(local);
        inner.inner(
          self
            .source
            .at_point(&global)
            .pullback(&cell_coords.linear_transform())
            .coeffs(),
          whitney.at_point(local).coeffs(),
        )
      };
      let value = self.qr.integrate_local(&inner_pointwise, geometry.vol());
      elvec[iwhitney] = value;
    }
    elvec
  }
}

#[cfg(test)]
mod test {
  use super::*;

  use ddf::whitney::lsf::WhitneyLsf;
  use manifold::{geometry::metric::simplex::SimplexLengths, topology::simplex::standard_subsimps};

  use approx::assert_relative_eq;

  #[test]
  fn hodge_mass0_is_scalar_mass() {
    for dim in 0..=3 {
      let geo = SimplexLengths::standard(dim);
      let hodge_mass = HodgeMassElmat::new(dim, 0).eval(&geo);
      let scalar_mass = scalar_mass_elmat(&geo);
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
            inner[(i, j)] = awhitney.inner(bwhitney, &geo.riemannian_metric());
          }
        }
        inner *= geo.vol();
        assert_relative_eq!(&difdif, &inner);
      }
    }
  }
}
