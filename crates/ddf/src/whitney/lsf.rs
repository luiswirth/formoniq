use {
  common::{
    combo::{factorialf, Combination, Sign},
    linalg::nalgebra::Matrix,
  },
  exterior::{field::ExteriorField, ExteriorGrade, MultiForm, MultiVector},
  manifold::{
    geometry::coord::{simplex::SimplexCoords, CoordRef},
    Dim,
  },
};

/// The Whitney local shape function $W_sigma$ of a DOF subsimplex.
///
/// Work in the formal barycentric space $Lambda(RR^(n+1))$, where the
/// vertex set $sigma$ is a blade $e_sigma$ and the barycentric coordinates
/// $lambda(x)$ are a vector. Then the Whitney form is the pullback along
/// the barycentric coordinate map of the Koszul contraction of the blade:
///
/// $W_sigma = k! med lambda^* (iota_(lambda(x)) e_sigma)
///   = k! sum_i (-1)^i lambda_(sigma_i)
///     dif lambda_(sigma_0) wedge dots.c hat(dif lambda_(sigma_i)) dots.c wedge dif lambda_(sigma_k)$
///
/// The contraction $iota_lambda$ is the Koszul operator $kappa$ of FEEC:
/// together with the pullback $lambda^*$ it generates the whole
/// $cal(P)_r^- Lambda^k$ family, of which the Whitney forms are the
/// lowest-order case.
#[derive(Debug, Clone)]
pub struct WhitneyLsf {
  cell_coords: SimplexCoords,
  /// The local vertex set of the DOF subsimplex.
  dof_simp: Combination,
  /// The differential of the barycentric coordinate map
  /// $lambda: RR^n -> RR^(n+1)$: the rows are the $dif lambda_i$.
  difbarys: Matrix,
}
impl WhitneyLsf {
  pub fn from_coords(cell_coords: SimplexCoords, dof_simp: Combination) -> Self {
    let difbarys = cell_coords.difbarys();
    Self {
      cell_coords,
      dof_simp,
      difbarys,
    }
  }
  pub fn standard(cell_dim: Dim, dof_simp: Combination) -> Self {
    Self::from_coords(SimplexCoords::standard(cell_dim), dof_simp)
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.dof_simp.card() - 1
  }

  /// The DOF vertex set as a blade in the formal barycentric space
  /// $Lambda^(k+1) (RR^(n+1))$.
  fn barycentric_blade(&self) -> MultiForm {
    let nvertices = self.cell_coords.nvertices();
    MultiForm::from_blade_signed(nvertices, Sign::Pos, self.dof_simp)
  }

  /// The constant exterior derivative of the Whitney LSF,
  /// $dif W_sigma = (k+1)! med lambda^* (e_sigma)
  /// = (k+1)! dif lambda_(sigma_0) wedge dots.c wedge dif lambda_(sigma_k)$.
  ///
  /// Vanishes automatically for the top grade, where $Lambda^(k+1) (RR^n)$
  /// is the zero space.
  pub fn dif(&self) -> MultiForm {
    factorialf(self.grade() + 1) * self.barycentric_blade().pullback(&self.difbarys)
  }
}

impl ExteriorField for WhitneyLsf {
  fn dim_ambient(&self) -> exterior::Dim {
    self.cell_coords.dim_ambient()
  }
  fn dim_intrinsic(&self) -> exterior::Dim {
    self.cell_coords.dim_intrinsic()
  }
  fn grade(&self) -> ExteriorGrade {
    self.grade()
  }
  fn at_point<'a>(&self, coord: impl Into<CoordRef<'a>>) -> MultiForm {
    let barys = MultiVector::line(self.cell_coords.global2bary(coord));
    let koszul = self.barycentric_blade().interior_product(&barys);
    factorialf(self.grade()) * koszul.pullback(&self.difbarys)
  }
}
